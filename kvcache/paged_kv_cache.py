import math
import time
import torch
import random
from torch import Tensor
import torch.nn.functional as F

from dataclasses import dataclass, field

# - Paged KV Cache
class BlockPool:
    def __init__(self, num_blocks: int, block_size: int, dim: int, device=None, dtype=torch.float32):
        self.k = torch.empty(num_blocks, block_size, dim, device=device, dtype=dtype)
        self.v = torch.empty(num_blocks, block_size, dim, device=device, dtype=dtype)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        # todo: ref count

    def alloc(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("no available blocks")
        block_id = self.free_blocks.pop()
        return block_id
    
    def release(self, block_id: int):
        # todo: check ref count of block_id
        self.free_blocks.append(block_id)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

# todo: preemption, 
class PagedKVCache:
    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.length = 0
        self.block_ids: list[int] = []
        self._k_view = None
        self._v_view = None

    def append(self, k, v):
        n = k.shape[0]
        slot = self.length % self.pool.block_size
        # pool.k[block_ids] is a copy (fancy indexing), not a view, so any write
        # below staleifies the cached gather regardless of whether a new block
        # was allocated.
        self._k_view = None
        self._v_view = None
        if n == 1:                                    # decode fast path
            if slot == 0:
                self.block_ids.append(self.pool.alloc())
            bid = self.block_ids[-1]
            self.pool.k[bid, slot] = k[0]
            self.pool.v[bid, slot] = v[0]
            self.length += 1
            return
        # prefill: fill block-by-block with slices instead of per-token
        i = 0
        while i < n:
            slot = self.length % self.pool.block_size
            if slot == 0:
                self.block_ids.append(self.pool.alloc())
            bid = self.block_ids[-1]
            take = min(self.pool.block_size - slot, n - i)
            self.pool.k[bid, slot:slot+take] = k[i:i+take]
            self.pool.v[bid, slot:slot+take] = v[i:i+take]
            self.length += take
            i += take

    def release(self):
        for block_id in self.block_ids:
            self.pool.release(block_id)
        self.length = 0

    def _gather(self, buf: Tensor):
        if self.length == 0:
            return buf.new_empty(0, buf.shape[-1])
        blocks = buf[self.block_ids]
        flat = blocks.reshape(-1, blocks.shape[-1])
        return flat[:self.length]
    
    @property
    def keys(self):
        if self._k_view is None or self._k_view.shape[0] < self.length:
            self._k_view = self.pool.k[self.block_ids].reshape(-1, self.pool.k.shape[-1])
        return self._k_view[:self.length]
    
    @property
    def values(self):
        if self._v_view is None or self._v_view.shape[0] < self.length:
            self._v_view = self.pool.v[self.block_ids].reshape(-1, self.pool.v.shape[-1])
        return self._v_view[:self.length]
    
    @property
    def num_blocks(self):
        return len(self.block_ids)

# - Contiguous Cache
class ContiguousKVCache:
    def __init__(self, max_len: int, dim: int, *, device=None, dtype=torch.float32):
        self._k = torch.empty(max_len, dim, device=device, dtype=dtype)
        self._v = torch.empty(max_len, dim, device=device, dtype=dtype)
        self.capacity = max_len
        self.length = 0

    def append(self, k: Tensor, v: Tensor):
        n = k.shape[0]
        if self.length + n > self.capacity:
            raise RuntimeError("ContiguousKVCache overflow")
        self._k[self.length : self.length + n] = k
        self._v[self.length : self.length + n] = v
        self.length += n

    @property
    def keys(self):
        return self._k[:self.length]

    @property
    def values(self):
        return self._v[:self.length]
    
# - Attention
class AttentionLayer:
    def __init__(self, dim: int, *, seed: int=42, device=None, dtype=torch.float32):
        self.dim = dim
        gen = torch.Generator(device=device).manual_seed(seed)
        scale = dim ** -0.5
        self.w_q = torch.randn(dim, dim, generator=gen, device=device, dtype=dtype) * scale
        self.w_k = torch.randn(dim, dim, generator=gen, device=device, dtype=dtype) * scale
        self.w_v = torch.randn(dim, dim, generator=gen, device=device, dtype=dtype) * scale
    
        self.w_qkv = torch.cat([self.w_q, self.w_k, self.w_v], dim=1)   # [dim, 3*dim]

    def project(self, x):
        return (x @ self.w_qkv).chunk(3, dim=-1)
    
    def sdpa(self, q: Tensor, k: Tensor, v: Tensor, *, is_causal: bool) -> Tensor:
        q4, k4, v4 = (t.unsqueeze(0).unsqueeze(0) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=is_causal)
        return out.squeeze(0).squeeze(0)

# - Scheduler
@dataclass
class Request:
    req_id: int
    prompt: Tensor
    max_decode_tokens: int
    done_len: int

@dataclass(eq=False)                          # identity equality: fields include tensors
class Sequence:
    req: Request
    cache: PagedKVCache | ContiguousKVCache | None # type: ignore
    generated: int
    last_token: Tensor
    history: Tensor | None = None
    generated_tokens: list[Tensor] = field(default_factory=list)

class ContinuousBatchScheduler:
    def __init__(self, layer: AttentionLayer, mode: str, budget_tokens: int, *, block_size: int = 16, max_batch: int = 64):
        assert mode in {"none", "contiguous", "paged"}
        self.layer = layer
        self.mode = mode
        self.budget_tokens = budget_tokens
        self.block_size = block_size
        self.max_batch = max_batch
        self.used_tokens = 0
        self.pool: BlockPool | None = None

        if mode == "paged":
            self.pool = BlockPool(
                num_blocks=budget_tokens // block_size,
                block_size=block_size,
                dim=layer.dim
            )

        return

    # -- admission

    def _reservation(self, req: Request) -> int: # type: ignore
        if self.mode == "contiguous":
            return req.prompt.shape[0] + req.max_decode_tokens
        elif self.mode == "paged":
            return math.ceil(req.prompt.shape[0] / self.block_size) * self.block_size
        return 0
    
    def _can_admit(self, req: Request) -> bool: # type: ignore
        if self.mode == "none":
            return True
        
        required = self._reservation(req)
        if self.mode == "paged":
            need = math.ceil(req.prompt.shape[0] / self.block_size)
            return need <= self.pool.num_free

        return self.used_tokens + required <= self.budget_tokens
    
    def _make_cache(self, req: Request):
        if self.mode == "paged":
            total = req.prompt.shape[0]
            return PagedKVCache(self.pool) # type: ignore
        if self.mode == "contiguous":
            self.used_tokens += req.prompt.shape[0] + req.max_decode_tokens
            return ContiguousKVCache(req.prompt.shape[0] + req.max_decode_tokens, self.layer.dim)
        
        return None
    
    def _retire(self, seq: Sequence) -> None:
        if self.mode == "paged":
            seq.cache.release() # type: ignore
        elif self.mode == "contiguous":
            self.used_tokens -= seq.req.prompt.shape[0] + seq.req.max_decode_tokens # type: ignore

    def _preempt(self, seq: Sequence) -> Request:
        seq.cache.release()
        if seq.generated == 0:
            resume_prompt = seq.req.prompt
        else:
            committed = seq.generated_tokens[:-1] # type: ignore
            resume_prompt = torch.cat([seq.req.prompt] + committed, dim=0)
        return Request(
            seq.req.req_id,
            resume_prompt,
            seq.req.max_decode_tokens,
            seq.req.done_len - seq.generated
        )

    # -- execution
    def _prefill(self, req: Request) -> Sequence:
        cache = self._make_cache(req)
        q, k, v = self.layer.project(req.prompt)
        if cache is None:
            out = self.layer.sdpa(q, k, v, is_causal=True)
            return Sequence(req, None, 0, out[-1:], history=req.prompt)

        cache.append(k, v)
        out = self.layer.sdpa(q, cache.keys, cache.values, is_causal=True)
        return Sequence(req, cache, 0, out[-1:], generated_tokens=[out[-1:]])
    
    def _decode_step(self, seq: Sequence) -> None:
        # todo: maybe add termination here
        if seq.cache is None:
            seq.history = torch.cat([seq.history, seq.last_token], dim=0)
            q, k, v = self.layer.project(seq.history)
            out = self.layer.sdpa(q, k, v, is_causal=True)
        else:
            q, k, v = self.layer.project(seq.last_token)
            seq.cache.append(k, v) # type: ignore
            out = self.layer.sdpa(q, seq.cache.keys, seq.cache.values, is_causal=False) # type: ignore
        seq.last_token = out[-1:]
        seq.generated_tokens.append(out[-1:])
        seq.generated += 1

    # -- main loop
    def run(self, requests: list[Request]) -> dict:
        pending = list(requests)
        running: list[Sequence] = []
        tokens_out: int = 0
        iterations: int = 0
        batch_sizes: list[int] = []
        preemptions: int = 0
        outputs: dict[int, list[Tensor]] = {}

        start = time.perf_counter()
        while pending or running:
            while pending and len(running) < self.max_batch and self._can_admit(pending[0]):
                running.append(self._prefill(pending.pop(0)))

            if not running:
                break

            batch_sizes.append(len(running))

            for seq in list(running):    # snapshot: running is mutated by preemption below
                if seq not in running:   # already evicted as another seq's victim this pass
                    continue
                while True:
                    try:
                        self._decode_step(seq)
                        break
                    except RuntimeError:                    # pool exhausted
                        victim = running[-1]                # LIFO: least invested
                        if victim is seq:                   # can't evict the one we're running
                            victim = running[-2] if len(running) > 1 else None
                        if victim is None:
                            raise RuntimeError("cannot make progress: pool too small")
                        running.remove(victim)
                        pending.insert(0, self._preempt(victim))
                        preemptions += 1
                outputs.setdefault(seq.req.req_id, []).append(seq.last_token)
                tokens_out += 1

            still_running = []
            for seq in running:
                if seq.generated >= seq.req.done_len:
                    self._retire(seq)
                else:
                    still_running.append(seq)

            running = still_running
            iterations += 1

        elapsed = time.perf_counter() - start
        return {
            "tokens_out": tokens_out,
            "iterations": iterations,
            "elapsed_s": elapsed,
            "avg_batch": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0,
            "outputs": {rid: torch.cat(toks, dim=0) for rid, toks in outputs.items()},
            "preemptions": preemptions
        }

def make_requests(n: int, dim: int, *, seed: int=42) -> list[Request]:
        rng = random.Random(seed)
        torch.manual_seed(seed)
        reqs = []
        
        for i in range(n):
            prompt_len = rng.randint(4, 16)
            done_len = rng.randint(200, 600)
            reqs.append(Request(i, torch.randn(prompt_len, dim), 1024, done_len))

        return reqs

def verify_consistency(results: dict[str, dict], baseline_mode: str = "none", *, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    # none recomputes every k/v from scratch each decode step while paged/contiguous
    # compute each once and reuse it, so float32 rounding accumulates along different
    # paths over hundreds of steps; a real logic bug shows up ~2 orders of magnitude
    # larger than this tolerance, not marginally over it.
    baseline = results[baseline_mode]["outputs"]
    for mode, result in results.items():
        if mode == baseline_mode:
            continue
        outputs = result["outputs"]
        assert outputs.keys() == baseline.keys(), f"{mode}: request set differs from {baseline_mode}"
        max_diff = 0.0
        for req_id, out in outputs.items():
            ref = baseline[req_id]
            assert out.shape == ref.shape, f"{mode} req {req_id}: shape {out.shape} != {ref.shape}"
            max_diff = max(max_diff, (out - ref).abs().max().item())
            if not torch.allclose(out, ref, atol=atol, rtol=rtol):
                raise AssertionError(f"{mode} req {req_id}: outputs diverge from {baseline_mode} (max abs diff {max_diff:.3e})")
        print(f"verified {mode} matches {baseline_mode} (max abs diff {max_diff:.3e})")

def print_results(results: dict) -> None:
    header = f"{'mode':<12} {'tokens':>8} {'iters':>7} {'time(s)':>9} {'tok/s':>9} {'avg batch':>10} {'preempt':>8}"
    print(header)
    print("-" * len(header))
    for mode, r in results.items():
        tok_s = r["tokens_out"] / r["elapsed_s"] if r["elapsed_s"] else 0.0
        print(
            f"{mode:<12} {r['tokens_out']:>8,} {r['iterations']:>7,} "
            f"{r['elapsed_s']:>9.2f} {tok_s:>9,.0f} {r['avg_batch']:>10.1f} "
            f"{r.get('preemptions', 0):>8,}"
        )

def main() -> None:
    dim = 256
    n_requests = 100
    budget_tokens = 5120

    layer = AttentionLayer(dim=dim)
    requests = make_requests(n_requests, dim)

    results = {}
    for mode in ("none", "contiguous", "paged"):
        scheduler = ContinuousBatchScheduler(layer, mode, budget_tokens, block_size=32)
        result = scheduler.run(requests)
        results[mode] = result
        
    print_results(results)
    verify_consistency(results)

if __name__ == "__main__":
    main()

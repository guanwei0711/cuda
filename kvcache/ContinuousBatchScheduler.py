import time

from typing import Union, Optional
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F

from RequestQueue import RequestQueue, Request
from ContiguousKVCache import ContiguousCachePool, ContiguousKVCache
from PagedKVCache import PagedCachePool, PagedKVCache

class AttentionLayer:
    def __init__(self, dim: int, device=None, dtype=torch.float32):
        scale = dim ** -0.5
        w_q = torch.randn(dim, dim, device=device, dtype=dtype) * scale
        w_k = torch.randn(dim, dim, device=device, dtype=dtype) * scale
        w_v = torch.randn(dim, dim, device=device, dtype=dtype) * scale
        self.w = torch.concat([w_q, w_k, w_v], dim=1)

    def project(self, x: Tensor):
        return (x @ self.w).chunk(3, dim=-1)

    def sdpa(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        q4, k4, v4 = (t.unsqueeze(0).unsqueeze(0) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=is_causal)
        return out.squeeze(0).squeeze(0)

@dataclass
class Sequence:
    req: Request
    generated: Tensor
    cache: Union[ContiguousKVCache, PagedKVCache, None] = None

@dataclass
class ContinuousBatchScheduler:
    pool: Union[ContiguousCachePool, PagedCachePool, None] = None
    
    def _try_add_request(self, req: Request) -> Optional[Sequence]:
        dim = req.prompt.shape[-1]
        if isinstance(self.pool, ContiguousCachePool):
            try:
                return Sequence(req, req.prompt.new_empty(0, dim), ContiguousKVCache(req.max_decode_tokens, self.pool))
            except RuntimeError: # no available space
                return None

        if isinstance(self.pool, PagedCachePool):
            try:
                return Sequence(req, req.prompt.new_empty(0, dim), PagedKVCache(self.pool, req.prompt.shape[0]))
            except RuntimeError: # no available space
                return None

        return Sequence(req, req.prompt.new_empty(0, dim))

    def _prefill(self, seq, attn_layer) -> Tensor:
        q, k, v = attn_layer.project(seq.req.prompt)
        out = attn_layer.sdpa(q, k, v, is_causal=True)
        if seq.cache:
            seq.cache.append(k, v)
        seq.generated = torch.cat([seq.generated, out[-1:]], dim=0)
        return out[-1:]

    def _decode(self, seq, attn_layer) -> Tensor:
        if seq.cache:
            q, k, v = attn_layer.project(seq.generated[-1:])
            seq.cache.append(k, v)
            out = attn_layer.sdpa(q, seq.cache.keys, seq.cache.values, is_causal=False)
        else:
            q, k, v = attn_layer.project(torch.concat([seq.req.prompt, seq.generated], dim=0))
            out = attn_layer.sdpa(q, k, v, is_causal=False)
        seq.generated = torch.cat([seq.generated, out[-1:]], dim=0)
        return out[-1:]

    def _retire(self, seq: Sequence):
        if seq.cache: seq.cache.release()

    def _preempt(self, seq: Sequence) -> Request:
        new_prompt = torch.concat([seq.req.prompt, seq.generated], dim=0)
        if seq.cache: seq.cache.release()
        return Request(seq.req.req_id, new_prompt, seq.req.max_decode_tokens, seq.req.terminal_len - seq.generated.shape[0])

    def run(self, max_batch: int, queue_capacity: int, attn_layer: AttentionLayer, requests: list[Request]) -> dict:
        req_queue = RequestQueue(capacity=queue_capacity)
        preempt_queue = RequestQueue(capacity=max_batch) if isinstance(self.pool, PagedCachePool) else None
        seq_batch: list[Sequence] = []

        batch_sizes: list[int] = []
        mem_utilization: list[float] = []
        tokens_out: int = 0
        iterations: int = 0
        preemptions: int = 0
        outputs: dict[int, list[Tensor]] = {}

        for request in requests: req_queue.enqueue(request)

        start = time.perf_counter()

        while req_queue or preempt_queue or seq_batch:
            while len(seq_batch) < max_batch and (preempt_queue or req_queue):
                if preempt_queue:
                    req = preempt_queue.dequeue()
                    seq = self._try_add_request(req)
                    if seq is None:
                        preempt_queue.requeue(req)
                        break
                elif req_queue:
                    req = req_queue.dequeue()
                    seq = self._try_add_request(req)
                    if seq is None:
                        req_queue.requeue(req)
                        break
                if seq:
                    tok = self._prefill(seq, attn_layer)
                    outputs.setdefault(seq.req.req_id, []).append(tok)
                    tokens_out += 1
                    seq_batch.append(seq)

            batch_sizes.append(len(seq_batch))
            if self.pool:
                live = sum(seq.cache.length for seq in seq_batch if seq.cache)
                mem_utilization.append(live / self.pool.capacity)

            preempt_victim_idx = len(seq_batch)
            for (idx, seq) in enumerate(seq_batch):
                if idx >= preempt_victim_idx:
                    break
                if isinstance(seq.cache, ContiguousKVCache) and seq.cache.length == seq.cache.max_length:
                    continue
                while True:
                    try:
                        tok = self._decode(seq, attn_layer)
                        outputs.setdefault(seq.req.req_id, []).append(tok)
                        tokens_out += 1
                        break
                    except RuntimeError: # page exhausted
                        preemptions += 1
                        preempt_victim_idx -= 1
                        victim = seq_batch[preempt_victim_idx]
                        if preempt_queue is not None: preempt_queue.enqueue(self._preempt(victim))
                        if idx == preempt_victim_idx: break

            next_seq: list[Sequence] = []
            for (idx, seq) in enumerate(seq_batch, start=0):
                if idx == preempt_victim_idx: break
                if isinstance(seq.cache, ContiguousKVCache) and seq.cache.length == seq.cache.max_length:
                    self._retire(seq)
                elif seq.generated.shape[0] >= seq.req.terminal_len:
                    self._retire(seq)
                else:
                    next_seq.append(seq)
            seq_batch = next_seq
            iterations += 1

        elapsed = time.perf_counter() - start
        return {
            "tokens_out": tokens_out,
            "iterations": iterations,
            "elapsed_s": elapsed,
            "avg_batch": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0,
            "outputs": {rid: torch.cat(toks, dim=0) for rid, toks in outputs.items()},
            "preemptions": preemptions,
            "mem_utilization": sum(mem_utilization) / len(mem_utilization) if mem_utilization else 0.0
        }


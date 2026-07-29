import torch
import random
import torch
import torch.nn.functional as F

from RequestQueue import Request
from ContiguousKVCache import ContiguousCachePool
from PagedKVCache import PagedCachePool
from ContinuousBatchScheduler import AttentionLayer, ContinuousBatchScheduler

def make_requests(n: int, dim: int, *, seed: int=42) -> list[Request]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    reqs = []
    
    for i in range(n):
        prompt_len = rng.randint(4, 16)
        done_len = rng.randint(200, 600)
        max_len = ((prompt_len + done_len) // 32 + 1) * 32
        reqs.append(Request(i, torch.randn(prompt_len, dim), max_len, done_len))

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
    header = f"{'mode':<12} {'tokens':>8} {'iters':>7} {'time(s)':>9} {'tok/s':>9} {'avg batch':>10} {'preempt':>8} {'mem_utilization':>16}"
    print(header)
    print("-" * len(header))
    for mode, r in results.items():
        tok_s = r["tokens_out"] / r["elapsed_s"] if r["elapsed_s"] else 0.0
        print(
            f"{mode:<12} {r['tokens_out']:>8,} {r['iterations']:>7,} "
            f"{r['elapsed_s']:>9.2f} {tok_s:>9,.0f} {r['avg_batch']:>10.1f} "
            f"{r.get('preemptions', 0):>8,} {r['mem_utilization']:>16.2f}"
        )

def main() -> None:
    dim = 256
    n_requests = 100
    budget_tokens = 4096

    requests = make_requests(n_requests, dim)
    layer = AttentionLayer(dim=dim)

    results = {}
    for mode in ["none", "contiguous", "paged"]:
    # for mode in ["paged"]:
        if mode == "contiguous": scheduler = ContinuousBatchScheduler(ContiguousCachePool(budget_tokens, dim))
        elif mode == "paged": scheduler = ContinuousBatchScheduler(PagedCachePool(budget_tokens // 32, 32, dim))
        else: scheduler = ContinuousBatchScheduler()
        result = scheduler.run(32, 1024, layer, requests)
        results[mode] = result
        
    print_results(results)
    verify_consistency(results)

if __name__ == "__main__":
    main()
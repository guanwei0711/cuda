# GEMM optimization process — from naive to near cuBLAS

<img width="796" height="186" alt="Screenshot 2026-07-04 at 2 23 57 PM" src="https://github.com/user-attachments/assets/bf5f07ab-0657-4541-bfd0-e0ee92e799e2" />

Profiling environment:
- Google Colab
- GPU: Nvidia Tesla T4 with SM frequency = 585 MHz (shared version)
- Calculating GEMM with C = alpha * A * B + beta * C, where A, B, and C all have dimension 2048 x 2048

## V1 — Naive summation over col / row per output cell

[v1_gemm_naive.cuh](gemm/v1_gemm_naive.cuh)

| Com. Throughput [%] | Mem. Throughput [%] | FP32 peak [%] | (Eligible / Active) warps per scheduler | Main Warp State[cycles] | Duration [ms] |
| - | - | - | - | - | - |
| 61.22 | 61.22 | 8 | 0.73 / 7.97 | LG Throttle(35.61) | 75.00 |

**Q: Compute / Memory / Latency bound?**

A: Latency bound.

**Q: Why?**

A: Eligible warps per scheduler is only 0.73, meaning the scheduler rarely has a ready warp — the LG throttle stalls aren't being hidden, which is the signature of a latency-bound kernel.

Further observations:
- Frequent global memory access leads to:
  1. Accumulation of global memory access latencies.
  2. Massive global memory accesses across warps, making the global memory pipeline congested, which leads to LG throttle and causes warp stalls.
  3. No other instructions available to hide the latency of global memory access.
- Global memory access is uncoalesced when accessing columns of matrix B.

Possible improvements:
- Reduce the frequency of global memory access, which both reduces accumulated latency and relieves pressure on the global memory pipeline.
  - Use shared memory tiling: load a tile of A and B from global memory into shared memory once, then have all threads in the block reuse those cached values — turning many redundant global loads into a few shared loads.

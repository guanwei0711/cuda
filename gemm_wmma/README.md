# WMMA GEMM optimization progress

Half-precision (fp16 in / fp32 accumulate) `C = alpha * A * B + beta * C` using the `nvcuda::wmma` tensor-core API, progressing from one warp computing one 16×16 tile with no shared memory, up through software-pipelined double buffering, and vectorized shared-memory access.

Profiling environment:
- Google Colab
- GPU: Nvidia Tesla T4 with SM frequency = 585 MHz (shared version)
- Compute Capability: 7.5
- Calculating GEMM with C = alpha * A * B + beta * C, where A, B, and C all have dimension 2048 x 2048

V7 reaches an 8.38× speedup over the naive V1 kernel (12.32 ms → 1.47 ms) and 64.63% of cuBLAS's throughput (0.95 ms) on a 2048×2048 fp16 GEMM.

## Overview

No bounds checking, M, N, K must be exact multiples of each kernel's block tile size — 16 for V1, 64 for V3, and 128 for V4–V7.
The DRAM roofline's ridge point sits at 75 OP/byte in this report.

## Table of Contents
- [V1 — Naive, No Shared Memory](#v1--naive-no-shared-memory)
- [V2 — Shared Memory Tiled](#v2--shared-memory-tiled)
- [V3 — 64×64 Warp Tiling](#v3--64x64-warp-tiling)
- [V4 — 128×128 Warp Tiling](#v4--128x128-warp-tiling)
- [V5 — Hoisted Fragment Loads (LDSM Prefetch)](#v5--hoisted-fragment-loads-ldsm-prefetch)
- [V6 — Double Buffering (Software Pipelining)](#v6--double-buffering-software-pipelining)
- [V7 — Vectorized Global Loads](#v7--vectorized-global-loads)
- [Summary: Progress Toward cuBLAS](#summary-progress-toward-cublas)

## V1 — Naive, No Shared Memory

[v1_gemm_naive.cu](v1_gemm_naive.cu)

Features:
- One warp (32 threads) per block, each block computes a single 16×16 output tile.
- `wmma::load_matrix_sync` reads A/B tile fragments directly from global memory every `k` step — no shared memory, no reuse across warps.

| Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Arithmetic Intensity [OP/byte] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - |
| 17.59 | 71.09 | 5.83 | 0.09 / 3.96 | LG Throttle | 6.12 | 12.32 | 7.71% |

**Q: Compute / Memory bound?**

A: Arithmetic intensity is left to the ridge point (75 OP/byte), so it's memory-bound.

Potential improvements:
- Stage A/B tiles through shared memory so all warps in a block reuse the same global loads.

## V2 — Shared Memory Tiled

[v2_gemm_smem_tiled.cu](v2_gemm_smem_tiled.cu)

Features:
- 4 warps/block, each warp owns one 16×16 WMMA tile within a 32×32 block tile.
- A/B tiles staged through shared memory (`tile_a`, `tile_b`) once per `k`-step and reused by all 4 warps.

| Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Arithmetic Intensity [OP/byte] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - |
| 62.93 | 62.93 | 12.43 | 0.74 / 7.82 | MIO Throttle | 16.48 | 5.76 | 16.49% |

**Q: Compute / Memory bound?**

A: Arithmetic intensity is left to the ridge point (75 OP/byte), so it's memory-bound.

Potential improvements:
- Each warp still only owns one WMMA tile; give each warp multiple tiles (register/warp-level tiling) to raise the arithmetic intensity.

## V3 — 64×64 Warp Tiling

[v3_gemm_m64n64k16.cu](v3_gemm_m64n64k16.cu)

Features:
- Block tile grows to 64×64; each of the 4 warps now computes a 2×2 grid of WMMA tiles (`M_TILES=2, N_TILES=2`) instead of just one.
- More `mma_sync` calls per shared-memory load, improving arithmetic intensity over V2.

| WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Arithmetic Intensity [OP/byte] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 2×2 | 60.44 | 60.44 | 19.63 | 0.41 / 6.49 | MIO Throttle | 43.04 | 3.65 | 26.03% |

**Q: Compute / Memory bound?**

A: Arithmetic intensity is left to the ridge point (75 OP/byte), so it's memory-bound.

Potential improvements:
- The kernel is still memory-bound; give each warp more tiles to increase arithmetic intensity.

## V4 — 128×128 Warp Tiling

[v4_gemm_m128n128k16.cu](v4_gemm_m128n128k16.cu)

Features:
- Same structure as V3, scaled up: block tile is 128×128, each warp computes a 4×4 grid of WMMA tiles (`M_TILES=4, N_TILES=4`) — 16 tiles per warp.

| WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Arithmetic Intensity [OP/byte] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 4×4 | 49.90 | 49.90 | 24.56 | 0.15 / 1.92 | Short Scoreboard | 156.74 | 2.92 | 32.53% |

**Q: Compute / Memory bound?**

A: Arithmetic intensity is right to the ridge point (75 OP/byte), so it's compute-bound.

Potential improvements:
- Hoist the fragment loads: each `a_frag[i]`/`b_frag[j]` is reused across the whole `(i, j)` double loop but currently gets reloaded from shared memory for every `(i, j)` pair.

## V5 — Hoisted Fragment Loads (LDSM Prefetch)

[v5_gemm_ldsm_prefetch.cu](v5_gemm_ldsm_prefetch.cu)

Features:
- `a_frag[M_TILES]` and `b_frag[N_TILES]` are now loaded **once per `k`-step** into arrays before the `(i, j)` `mma_sync` loop, instead of being reloaded inside it — removes the redundant `load_matrix_sync` calls V4 had.

| WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Arithmetic Intensity [OP/byte] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 4×4 | 44.65 | 44.65 | 34.84 | 0.20 / 1.92 | Wait / MIO Throttle | 177.99 | 1.95 | 48.72% |

**Q: Compute / Memory bound?**

A: Arithmetic intensity is right to the ridge point (75 OP/byte), so it's compute-bound.

Potential improvements:
- Global-to-shared-memory loads still fully block compute at the top of every `k`-step; overlap them with the previous step's compute via double buffering.

## V6 — Double Buffering (Software Pipelining)

[v6_gemm_double_buffer.cu](v6_gemm_double_buffer.cu)

Features:
- Two-stage (`[2]`) shared-memory buffers for `tile_a`/`tile_b`. While computing on the current buffer, the next `k`-tile is loaded from global memory into per-thread staging registers (`stage_a`/`stage_b`), then written into the other buffer — overlapping global memory latency with tensor-core compute.
- Loads are still scalar (one `half` element per thread per transaction).

| WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Arithmetic Intensity [OP/byte] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 4×4 | 50.30 | 50.30 | 39.25 | 0.23 / 1.91 | Wait / MIO Throttle | 173.54 | 1.82 | 52.20% |

**Q: Compute / Memory bound?**

A: Arithmetic intensity is right to the ridge point (75 OP/byte), so it's compute-bound.

Potential improvements:
- Vectorize the global loads (currently one scalar `half` per thread) to cut the instruction count.

## V7 — Vectorized Global Loads

[v7_gemm_vectorization.cu](v7_gemm_vectorization.cu)

Features:
- Same double-buffered pipeline as V6, but both the initial load and the prefetch stage use `float4` (128-bit, 8 `half`s at a time) loads via `FLOAT4`/`CFLOAT4` instead of scalar element-wise loads — fewer, wider memory transactions.

| WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Arithmetic Intensity [OP/byte] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 4×4 | 50.39 | 40.74 | 48.75 | 0.17 / 1.89 | Wait / MIO Throttle | 155.11 | 1.47 | 64.63% |

**Q: Compute / Memory bound?**

A: Arithmetic intensity is right to the ridge point (75 OP/byte), so it's compute-bound.

## Summary: Progress Toward cuBLAS

| Version | Block Tile | Key Technique Added | Duration [ms] | Tensor Pipe Util. [%] | Speedup vs V1 | % of cuBLAS |
| - | - | - | - | - | - | - |
| V1 — Naive | 16×16 | — | 12.32 | 5.83 | 1.00x | 7.71% |
| V2 — Shared Memory Tiled | 32×32 | Shared-memory tile reuse + bounds checking | 5.76 | 12.43 | 2.14x | 16.49% |
| V3 — 64×64 Warp Tiling | 64×64 | Multiple WMMA tiles/warp | 3.65 | 19.63 | 3.38x | 26.03% |
| V4 — 128×128 Warp Tiling | 128×128 | Larger warp tile | 2.92 | 24.56 | 4.22x | 32.53% |
| V5 — LDSM Prefetch | 128×128 | Hoisted fragment loads | 1.95 | 34.84 | 6.32x | 48.72% |
| V6 — Double Buffer | 128×128 | Software-pipelined global loads | 1.82 | 39.25 | 6.77x | 52.20% |
| V7 — Vectorization | 128×128 | `float4` global loads | 1.47 | 48.75 | 8.38x | 64.63% |
| **cuBLAS** (`cublasGemmEx`, tensor-op) | 128×128 | — | 0.95 | 74.86 | 12.97x | 100% |

## Future Improvements
- `load_matrix_sync` currently issues `LDSM.16.M88.2` instead of `LDSM.16.M88.4`, doubling the instruction count. Using `ldmatrix` combined with `mma.sync` could load 4 tiles in a single instruction instead.
- To avoid the bank conflicts introduced by `load_matrix_sync`, shared memory is padded — but that padding causes bank conflicts on the vectorized loads instead. Combining the `ldmatrix` + `mma.sync` approach above with swizzling may avoid both.

# WMMA GEMM optimization progress — from naive tensor cores to near cuBLAS

Half-precision (fp16 in / fp32 accumulate) `C = alpha * A * B + beta * C` using the `nvcuda::wmma` tensor-core API, progressing from one warp computing one 16×16 tile with no shared memory, up through warp-level register tiling, prefetched fragment loads, software-pipelined double buffering, and vectorized shared-memory access.

Profiling environment:
- Google Colab
- GPU: Nvidia Tesla T4 with SM frequency = 585 MHz (shared version)
- Compute Capability: 7.5
- Calculating GEMM with C = alpha * A * B + beta * C, where A, B, and C all have dimension 2048 x 2048

_(fill in: one-line summary of the final speedup / % of cuBLAS reached, once measured)_

## Build & Run

```
nvcc gemm_wmma/main.cu -lcublas -o gemm_wmma/main
./gemm_wmma/main [DIM=2048] [run_cublas=0|1]
```

`main.cu` runs each kernel below against a CPU fp32 reference (inputs are round-tripped through `half` first, so the reference sees the same quantized values the GPU does) and prints max-abs-error. Correctness checking is skipped above 1024×1024 to keep the CPU reference fast; pass `1` as the second argument to also run a `cublasGemmEx` tensor-core reference. No timing/benchmarking code is included yet — the tables below are templates to fill in from Nsight Compute or your own timing harness.

## Table of Contents
- [V1 — Naive, No Shared Memory](#v1--naive-no-shared-memory)
- [V2 — Shared Memory Tiled](#v2--shared-memory-tiled)
- [V3 — 64×64 Warp Tiling](#v3--64x64-warp-tiling)
- [V4 — 128×128 Warp Tiling](#v4--128x128-warp-tiling)
- [V5 — Hoisted Fragment Loads (LDSM Prefetch)](#v5--hoisted-fragment-loads-ldsm-prefetch)
- [V6 — Double Buffering (Software Pipelining)](#v6--double-buffering-software-pipelining)
- [V7 — Vectorized Global Loads](#v7--vectorized-global-loads)
- [V0 — Playground (experimental, not wired into main)](#v0--playground-experimental-not-wired-into-main)
- [Summary: Progress Toward cuBLAS](#summary-progress-toward-cublas)

## V1 — Naive, No Shared Memory

[v1_gemm_naive.cu](v1_gemm_naive.cu)

Features:
- One warp (32 threads) per block, each block computes a single 16×16 output tile.
- `wmma::load_matrix_sync` reads A/B tile fragments directly from global memory every `k` step — no shared memory, no reuse across warps.
- No bounds checking: M, N, K must all be exact multiples of 16.

| Block Tile | K-Tile | Warps/Block | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - | - |
| 16×16 | 16 | 1 | — | — | — | — | — | — | — |

**Q: Compute / Memory / Latency bound?**

A: _(fill in)_

Potential improvements:
- Stage A/B tiles through shared memory so all warps in a block reuse the same global loads.

## V2 — Shared Memory Tiled

[v2_gemm_smem_tiled.cu](v2_gemm_smem_tiled.cu)

Features:
- 4 warps/block, each warp owns one 16×16 WMMA tile within a 32×32 block tile.
- A/B tiles staged through shared memory (`tile_a`, `tile_b`) once per `k`-step and reused by all 4 warps.
- The only kernel in this set with bounds checking on every load/store — safe for M/N/K that aren't multiples of the tile size (zero-padded loads, clamped stores).

| Block Tile | K-Tile | Warps/Block | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Eligible/Active warps per scheduler | Main Warp State | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - | - |
| 32×32 | 16 | 4 | — | — | — | — | — | — | — |

**Q: Compute / Memory / Latency bound?**

A: _(fill in)_

Potential improvements:
- Each warp still only owns one WMMA tile; give each warp multiple tiles (register/warp-level tiling) to raise the arithmetic-to-load ratio.

## V3 — 64×64 Warp Tiling

[v3_gemm_m64n64k16.cu](v3_gemm_m64n64k16.cu)

Features:
- Block tile grows to 64×64; each of the 4 warps now computes a 2×2 grid of WMMA tiles (`M_TILES=2, N_TILES=2`) instead of just one.
- More `mma_sync` calls per shared-memory load, improving compute-to-memory ratio over V2.
- No bounds checking — M/N/K must be exact multiples of the tile size.

| Block Tile | K-Tile | Warps/Block | WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 64×64 | 16 | 4 | 2×2 | — | — | — | — | — |

**Q: Compute / Memory / Latency bound?**

A: _(fill in)_

## V4 — 128×128 Warp Tiling

[v4_gemm_m128n128k16.cu](v4_gemm_m128n128k16.cu)

Features:
- Same structure as V3, scaled up: block tile is 128×128, each warp computes a 4×4 grid of WMMA tiles (`M_TILES=4, N_TILES=4`) — 16 tiles per warp.
- Inside the `(i, j)` loop, `a_frag`/`b_frag` are still reloaded from shared memory on every iteration, including redundant reloads of the same fragment across the inner loop.

| Block Tile | K-Tile | Warps/Block | WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 128×128 | 16 | 4 | 4×4 | — | — | — | — | — |

**Q: Compute / Memory / Latency bound?**

A: _(fill in)_

Potential improvements:
- Hoist the fragment loads: each `a_frag[i]`/`b_frag[j]` is reused across the whole `(i, j)` double loop but currently gets reloaded from shared memory for every `(i, j)` pair.

## V5 — Hoisted Fragment Loads (LDSM Prefetch)

[v5_gemm_ldsm_prefetch.cu](v5_gemm_ldsm_prefetch.cu)

Features:
- Same 128×128 block tile / 4×4 warp tiling as V4.
- `a_frag[M_TILES]` and `b_frag[N_TILES]` are now loaded **once per `k`-step** into arrays before the `(i, j)` `mma_sync` loop, instead of being reloaded inside it — removes the redundant `load_matrix_sync` calls V4 had.

| Block Tile | K-Tile | Warps/Block | WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 128×128 | 16 | 4 | 4×4 | — | — | — | — | — |

**Q: Compute / Memory / Latency bound?**

A: _(fill in)_

Potential improvements:
- Global-to-shared-memory loads still fully block compute at the top of every `k`-step; overlap them with the previous step's compute via double buffering.

## V6 — Double Buffering (Software Pipelining)

[v6_gemm_double_buffer.cu](v6_gemm_double_buffer.cu)

Features:
- Same 128×128 / 4×4 warp tiling as V4/V5.
- Two-stage (`[2]`) shared-memory buffers for `tile_a`/`tile_b`. While computing on the current buffer, the next `k`-tile is loaded from global memory into per-thread staging registers (`stage_a`/`stage_b`), then written into the other buffer — overlapping global memory latency with tensor-core compute.
- Loads are still scalar (one `half` element per thread per transaction).

| Block Tile | K-Tile | Warps/Block | WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 128×128 | 16 | 4 | 4×4 | — | — | — | — | — |

**Q: Compute / Memory / Latency bound?**

A: _(fill in)_

Potential improvements:
- Vectorize the global loads (currently one scalar `half` per thread) to cut the instruction count for the prefetch stage.

## V7 — Vectorized Global Loads

[v7_gemm_vectorization.cu](v7_gemm_vectorization.cu)

Features:
- Same double-buffered pipeline as V6, but both the initial load and the prefetch stage use `float4` (128-bit, 8 `half`s at a time) loads via `FLOAT4`/`CFLOAT4` instead of scalar element-wise loads — fewer, wider memory transactions.

| Block Tile | K-Tile | Warps/Block | WMMA Tiles/Warp | Compute Throughput [%] | Mem. Throughput [%] | Tensor Pipe Util. [%] | Duration [ms] | % of cuBLAS |
| - | - | - | - | - | - | - | - | - |
| 128×128 | 16 | 4 | 4×4 | — | — | — | — | — |

**Q: Compute / Memory / Latency bound?**

A: _(fill in)_

## Summary: Progress Toward cuBLAS

| Version | Block Tile | Key Technique Added | Duration [ms] | Tensor Pipe Util. [%] | Speedup vs V1 | % of cuBLAS |
| - | - | - | - | - | - | - |
| V1 — Naive | 16×16 | — | — | — | 1.00x | — |
| V2 — Shared Memory Tiled | 32×32 | Shared-memory tile reuse + bounds checking | — | — | — | — |
| V3 — 64×64 Warp Tiling | 64×64 | Multiple WMMA tiles/warp | — | — | — | — |
| V4 — 128×128 Warp Tiling | 128×128 | Larger warp tile | — | — | — | — |
| V5 — LDSM Prefetch | 128×128 | Hoisted fragment loads | — | — | — | — |
| V6 — Double Buffer | 128×128 | Software-pipelined global loads | — | — | — | — |
| V7 — Vectorization | 128×128 | `float4` global loads | — | — | — | — |
| **cuBLAS** (`cublasGemmEx`, tensor-op) | — | — | — | — | — | 100% |

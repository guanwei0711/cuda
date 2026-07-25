#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

namespace v7_dims {
    static constexpr int WARPS = 4;
    static constexpr int WARP_DIM_X = 2;
    static constexpr int WARP_DIM_Y = WARPS / WARP_DIM_X;
    static constexpr int WARP_SIZE = 32;

    static constexpr int WMMA_M = 16;
    static constexpr int M_TILES = 4;
    static constexpr int M_SMEM_ROWS = WARP_DIM_Y * WMMA_M * M_TILES;

    static constexpr int WMMA_N = 16;
    static constexpr int N_TILES = 4;
    static constexpr int N_SMEM_COLS = WARP_DIM_X * WMMA_N * N_TILES;
}

__global__ void v7_gemm_gmem_pipelining(const half* A, const half* B, half* C,
                               int M, int N, int K, float alpha, float beta) {
    using namespace v7_dims;   

    int g_tile_id = 0;
    __shared__ half tile_a[2][M_SMEM_ROWS][WMMA_M + 8];
    __shared__ half tile_b[2][WMMA_N][N_SMEM_COLS + 8];

    int tid = threadIdx.x;
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int row = block_m * M_SMEM_ROWS;
    int col = block_n * N_SMEM_COLS;
    
    constexpr int a_dim_x = WMMA_M, a_dim_y = (WARP_SIZE * WARPS) / a_dim_x;
    int a_thread_x = tid % a_dim_x;
    int a_thread_y = tid / a_dim_x;

    constexpr int b_dim_x = N_SMEM_COLS, b_dim_y = (WARP_SIZE * WARPS) / b_dim_x;
    int b_thread_x = tid % b_dim_x;
    int b_thread_y = tid / b_dim_x;

    // which 16x16 block current in 32x32 block current warp hold
    int c_warp_x = warp_id % WARP_DIM_X;
    int c_warp_y = warp_id / WARP_DIM_X;
    int tile_warp_row = c_warp_y * WMMA_M;
    int tile_warp_col = c_warp_x * WMMA_N;

    int tile_id = 0;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[M_TILES * N_TILES];
    for (int i = 0; i < M_TILES * N_TILES; ++i) wmma::fill_fragment(acc_frag[i], 0.f);

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    half stage_a[M_SMEM_ROWS / a_dim_y];
    half stage_b[WMMA_N / b_dim_y];

    #pragma unroll
    for (int i = 0; i < M_SMEM_ROWS; i += a_dim_y) {
        int arow = row + i + a_thread_y;
        int acol = a_thread_x;
        tile_a[0][i + a_thread_y][a_thread_x] = A[arow * K + acol];
    }

    #pragma unroll
    for (int i = 0; i < WMMA_N; i += b_dim_y) {
        int brow = i + b_thread_y;
        int bcol = col + b_thread_x;
        tile_b[0][i + b_thread_y][b_thread_x] = B[brow * N + bcol];
    }
    __syncthreads();
    
    for (int k = 0; k < K; k += 16) {
        // load a tile
        if (k + 16 < K) {
            int stage_a_id = 0;
            #pragma unroll
            for (int i = 0; i < M_SMEM_ROWS; i += a_dim_y) {
                int arow = row + i + a_thread_y;
                int acol = k + 16 + a_thread_x;
                stage_a[stage_a_id++]= A[arow * K + acol];
                // tile_a[g_tile_id ^ 1][i + a_thread_y][a_thread_x] 
            }
            
            int stage_b_id = 0;
            #pragma unroll
            for (int i = 0; i < WMMA_N; i += b_dim_y) {
                int brow = i + k + 16 + b_thread_y;
                int bcol = col + b_thread_x;
                stage_b[stage_b_id++] = B[brow * N + bcol];
                // tile_b[g_tile_id ^ 1][i + b_thread_y][b_thread_x] = B[brow * N + bcol];
            }
        }
        wmma::load_matrix_sync(a_frag[0], &tile_a[g_tile_id][tile_warp_row][0], WMMA_M + 8);
        wmma::load_matrix_sync(b_frag[0], &tile_b[g_tile_id][0][tile_warp_col], N_SMEM_COLS + 8);

        #pragma unroll
        for (int t = 0; t < M_TILES * N_TILES; ++t) {
            int i = t / N_TILES, j = t % N_TILES;
            if (t < M_TILES * N_TILES - 1) {
                int advi = (t + 1) / N_TILES, advj = (t + 1) % N_TILES;
                wmma::load_matrix_sync(a_frag[tile_id ^ 1], &tile_a[g_tile_id][tile_warp_row + advi * WARP_DIM_Y * WMMA_M][0], WMMA_M + 8);
                wmma::load_matrix_sync(b_frag[tile_id ^ 1], &tile_b[g_tile_id][0][tile_warp_col + advj * WARP_DIM_X * WMMA_N], N_SMEM_COLS + 8);
            } else if (k + 16 < K) {
                int stage_a_id = 0;
                #pragma unroll
                for (int i = 0; i < M_SMEM_ROWS; i += a_dim_y) {
                    tile_a[g_tile_id ^ 1][i + a_thread_y][a_thread_x] = stage_a[stage_a_id++];
                }
                
                int stage_b_id = 0;
                #pragma unroll
                for (int i = 0; i < WMMA_N; i += b_dim_y) {
                    tile_b[g_tile_id ^ 1][i + b_thread_y][b_thread_x] = stage_b[stage_b_id++];
                }
            }
            wmma::mma_sync(acc_frag[i * N_TILES + j], a_frag[tile_id], b_frag[tile_id], acc_frag[i * N_TILES + j]);
            tile_id ^= 1;
        }
        __syncthreads();
        g_tile_id ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < M_TILES; ++i) {    
        #pragma unroll
        for (int j = 0; j < N_TILES; ++j) {
            wmma::load_matrix_sync(c_frag, &C[(row + tile_warp_row + i * WARP_DIM_Y * WMMA_M) * N + (col + tile_warp_col + j * WARP_DIM_X * WMMA_N)], N, wmma::mem_row_major);
            for (int t = 0; t < acc_frag[i * N_TILES + j].num_elements; ++t) {
                c_frag.x[t] = __float2half(alpha * acc_frag[i * N_TILES + j].x[t] + beta * __half2float(c_frag.x[t]));
            }
            wmma::store_matrix_sync(&C[(row + tile_warp_row + i * WARP_DIM_Y * WMMA_M) * N + (col + tile_warp_col + j * WARP_DIM_X * WMMA_N)], c_frag, N, wmma::mem_row_major);
        }
    }
}
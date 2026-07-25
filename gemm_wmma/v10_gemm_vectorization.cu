#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

namespace v10_dims {
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

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CFLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

__global__ void v10_gemm_vectorization(const half* A, const half* B, half* C,
                               int M, int N, int K, float alpha, float beta) {
    using namespace v10_dims;   

    __shared__ half tile_a[2][M_SMEM_ROWS][WMMA_M + 8];
    __shared__ half tile_b[2][WMMA_N][N_SMEM_COLS + 8];
    int g_tile_id = 0;
    
    int tid = threadIdx.x;
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int row = block_m * M_SMEM_ROWS;
    int col = block_n * N_SMEM_COLS;
    
    constexpr int a_dim_x = WMMA_M / A_VEC_SIZE, a_dim_y = (WARP_SIZE * WARPS) / a_dim_x;
    int a_thread_x = tid % a_dim_x;
    int a_thread_y = tid / a_dim_x;

    constexpr int b_dim_x = N_SMEM_COLS / B_VEC_SIZE, b_dim_y = (WARP_SIZE * WARPS) / b_dim_x;
    int b_thread_x = tid % b_dim_x;
    int b_thread_y = tid / b_dim_x;

    #pragma unroll
    for (int i = 0; i < M_SMEM_ROWS; i += a_dim_y) {
        int arow = row + i + a_thread_y;
        int acol = a_thread_x * A_VEC_SIZE;
        FLOAT4(tile_a[0][i + a_thread_y][a_thread_x * A_VEC_SIZE]) = CFLOAT4(A[arow * K + acol]);
    }

    #pragma unroll
    for (int i = 0; i < WMMA_N; i += b_dim_y) {
        int brow = i + b_thread_y;
        int bcol = col + b_thread_x * B_VEC_SIZE;
        FLOAT4(tile_b[0][i + b_thread_y][b_thread_x * B_VEC_SIZE]) = CFLOAT4(B[brow * N + bcol]);
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[M_TILES];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[N_TILES];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[M_TILES * N_TILES];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    #pragma unroll
    for (int i = 0; i < M_TILES * N_TILES; ++i) wmma::fill_fragment(acc_frag[i], 0.f);

    // which 16x16 block current in 32x32 block current warp hold
    int c_warp_x = warp_id % WARP_DIM_X;
    int c_warp_y = warp_id / WARP_DIM_X;
    int tile_warp_row = c_warp_y * WMMA_M;
    int tile_warp_col = c_warp_x * WMMA_N;
    half stage_a[M_SMEM_ROWS / a_dim_y * A_VEC_SIZE];
    half stage_b[WMMA_N / b_dim_y * B_VEC_SIZE];

    __syncthreads();
    
    for (int k = 0; k < K; k += 16) {
        if (k + 16 < K) {
            #pragma unroll
            for (int i = 0; i < M_SMEM_ROWS; i += a_dim_y) {
                int arow = row + i + a_thread_y;
                int acol = k + 16 + a_thread_x * A_VEC_SIZE;
                FLOAT4(stage_a[(i / a_dim_y) * A_VEC_SIZE]) = CFLOAT4(A[arow * K + acol]);
                // tile_a[g_tile_id ^ 1][i + a_thread_y][a_thread_x] = A[arow * K + acol];
            }

            #pragma unroll
            for (int i = 0; i < WMMA_N; i += b_dim_y) {
                int brow = i + k + 16 + b_thread_y;
                int bcol = col + b_thread_x * B_VEC_SIZE;
                FLOAT4(stage_b[(i / b_dim_y) * B_VEC_SIZE]) = CFLOAT4(B[brow * N + bcol]);
                // tile_b[g_tile_id ^ 1][i + b_thread_y][b_thread_x] = B[brow * N + bcol];
            }
        }

        for (int i = 0; i < M_TILES; ++i) wmma::load_matrix_sync(a_frag[i], &tile_a[g_tile_id][tile_warp_row + i * WARP_DIM_Y * WMMA_M][0], WMMA_M + 8);
        for (int j = 0; j < N_TILES; ++j) wmma::load_matrix_sync(b_frag[j], &tile_b[g_tile_id][0][tile_warp_col + j * WARP_DIM_X * WMMA_N], N_SMEM_COLS + 8);

        #pragma unroll
        for (int i = 0; i < M_TILES; ++i) {    
            #pragma unroll
            for (int j = 0; j < N_TILES; ++j) {
                wmma::mma_sync(acc_frag[i * N_TILES + j], a_frag[i], b_frag[j], acc_frag[i * N_TILES + j]);
            }
        }
        if (k + 16 < K) {
            #pragma unroll
            for (int i = 0; i < M_SMEM_ROWS; i += a_dim_y) {
                int arow = row + i + a_thread_y;
                int acol = k + 16 + a_thread_x * A_VEC_SIZE;
                FLOAT4(tile_a[g_tile_id ^ 1][i + a_thread_y][a_thread_x * A_VEC_SIZE]) = FLOAT4(stage_a[i / a_dim_y * A_VEC_SIZE]);
            }

            #pragma unroll
            for (int i = 0; i < WMMA_N; i += b_dim_y) {
                int brow = i + k + 16 + b_thread_y;
                int bcol = col + b_thread_x * B_VEC_SIZE;
                FLOAT4(tile_b[g_tile_id ^ 1][i + b_thread_y][b_thread_x * B_VEC_SIZE]) = FLOAT4(stage_b[i / b_dim_y * B_VEC_SIZE]);
            }
            g_tile_id ^= 1;
            __syncthreads();
        }
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
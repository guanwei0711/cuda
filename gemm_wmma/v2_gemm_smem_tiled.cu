#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

#define WARPS 4
#define WARP_DIM_X 2
#define WARP_DIM_Y (WARPS / WARP_DIM_X)
#define WARP_SIZE 32

#define WMMA_M 16
#define M_SMEM_ROWS (WARP_DIM_Y * WMMA_M)

#define WMMA_N 16
#define N_SMEM_COLS (WARP_DIM_X * WMMA_N)

__global__ void v2_gemm_smem_tiled(const half* A, const half* B, half* C,
                               int M, int N, int K, float alpha, float beta) {

    __shared__ half tile_a[M_SMEM_ROWS][WMMA_M + 8];
    __shared__ half tile_b[WMMA_N][N_SMEM_COLS + 8];
    __shared__ float scratch[M_SMEM_ROWS][N_SMEM_COLS + 8];

    int tid = threadIdx.x;
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int row = block_m * M_SMEM_ROWS;
    int col = block_n * N_SMEM_COLS;
    
    constexpr int a_dim_x = WMMA_M, a_dim_y = (WARP_SIZE * WARPS) / a_dim_x;
    int a_thread_x = tid % a_dim_x;
    int a_thread_y = tid / a_dim_x;

    constexpr int b_dim_x = N_SMEM_COLS, b_dim_y = (WARP_SIZE * WARPS) / b_dim_x;
    int b_thread_x = tid % b_dim_x;
    int b_thread_y = tid / b_dim_x;

    int c_warp_x = warp_id % WARP_DIM_X;
    int c_warp_y = warp_id / WARP_DIM_X;
    int tile_warp_row = c_warp_y * WMMA_M;
    int tile_warp_col = c_warp_x * WMMA_N;

    constexpr int c_dim_x = WMMA_N, c_dim_y = 32 / c_dim_x;
    int c_thread_x = lane_id % c_dim_x;
    int c_thread_y = lane_id / c_dim_x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.f);

    for (int k = 0; k < K; k += 16) {
        // load a tile
        #pragma unroll
        for (int i = 0; i < M_SMEM_ROWS; i += a_dim_y) {
            int arow = row + i + a_thread_y;
            int acol = k + a_thread_x;
            tile_a[i + a_thread_y][a_thread_x] = arow < M && acol < K ? A[arow * K + acol] : half{0.0};
        }

        #pragma unroll
        for (int i = 0; i < WMMA_N; i += b_dim_y) {
            int brow = i + k + b_thread_y;
            int bcol = col + b_thread_x;
            tile_b[i + b_thread_y][b_thread_x] = brow < K && bcol < N ? B[brow * N + bcol] : half{0.0};
        }
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag, &tile_a[tile_warp_row][0], WMMA_M + 8);
        wmma::load_matrix_sync(b_frag, &tile_b[0][tile_warp_col], N_SMEM_COLS + 8);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }
    
    wmma::store_matrix_sync(&scratch[tile_warp_row][tile_warp_col],
                            acc_frag, N_SMEM_COLS + 8, wmma::mem_row_major);

    #pragma unroll
    for (int i = 0; i < WMMA_M; i += c_dim_y) {
        int r = row + tile_warp_row + c_thread_y + i;
        int c = col + tile_warp_col + c_thread_x;
        if (r < M && c < N)
            C[r * N + c] = __float2half(
                alpha * scratch[tile_warp_row + c_thread_y + i][tile_warp_col + c_thread_x] + beta * __half2float(C[r * N + c])
            );
    }
}
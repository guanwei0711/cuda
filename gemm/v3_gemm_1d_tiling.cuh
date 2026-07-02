#pragma once
#include <cuda_runtime.h>

template<int Bm = 64, int Bn = 64, int Bk = 8, int Tm = 16, int THREADS = 256>
__global__ void v3_gemm_1d_tiling(const float* __restrict__ A, const float* __restrict__ B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ float tile_a[Bm][Bk];
    __shared__ float tile_b[Bk][Bn];
    int tid = threadIdx.x;
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    constexpr int a_dim_x = Bk, a_dim_y = THREADS / a_dim_x;
    constexpr int b_dim_y = Bk, b_dim_x = THREADS / b_dim_y;
    constexpr int c_dim_x = Bn, c_dim_y = THREADS / c_dim_x;

    int a_thread_y = tid / a_dim_x;
    int a_thread_x = tid % a_dim_x;

    int b_thread_y = tid / b_dim_x;
    int b_thread_x = tid % b_dim_x;

    int c_thread_y = tid / c_dim_x;
    int c_thread_x = tid % c_dim_x;
    float thread_results[Tm] = { 0.0f };

    for (int k = 0; k < K; k += Bk) {
        // step1 load into shared tile
        for (int i = a_thread_y; i < Bm; i += a_dim_y) {
            int row = r0 + i;
            for (int j = a_thread_x; j < Bk; j += a_dim_x) {
                int col = k + j;
                tile_a[i][j] = row < M && col < K ? A[row * K + col] : 0.0f;
            }
        }

        for (int i = b_thread_y; i < Bk; i += b_dim_y) {
            int row = k + i;
            for (int j = b_thread_x; j < Bn; j += b_dim_x) {
                int col = c0 + j;
                tile_b[i][j] = row < K && col < N ? B[row * N + col] : 0.0f;
            }
        }
        __syncthreads();
        
        for (int i = 0; i < Bk; ++i) {
            float regB = tile_b[i][c_thread_x];
            for (int j = 0; j < Tm; ++j) {
                thread_results[j] += regB * tile_a[c_thread_y * Tm + j][i];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < Tm; ++i) {
        int row = r0 + c_thread_y * Tm + i;
        int col = c0 + c_thread_x;
        if (row < M && col < N) C[row * N + col] = alpha * thread_results[i] + beta * C[row * N + col];
    }
}
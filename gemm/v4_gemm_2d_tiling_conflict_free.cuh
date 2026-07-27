#pragma once
#include <cuda_runtime.h>

// Bm * Bk % THREADS == 0
// Bn * Bk % THREADS == 0
// (Bm / Tm) * (Bn / Tn) == THREADS
template<int Bm = 64, int Bn = 64, int Bk = 4, int Tm = 4, int Tn = 4, int THREADS = 256>
__global__ void v4_gemm_2d_tiling_conflict_free(const float* __restrict__ A, const float* __restrict__ B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ float tile_a[Bk][Bm];
    __shared__ float tile_b[Bk][Bn];
    int tid = threadIdx.x;
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    constexpr int a_dim_x = Bk, a_dim_y = THREADS / a_dim_x;
    constexpr int b_dim_x = 32, b_dim_y = THREADS / b_dim_x;
    constexpr int c_dim_x = Bn / Tn, c_dim_y = THREADS / c_dim_x;

    int a_thread_y = tid / a_dim_x;
    int a_thread_x = tid % a_dim_x;

    int b_thread_y = tid / b_dim_x;
    int b_thread_x = tid % b_dim_x;

    int c_thread_y = tid / c_dim_x;
    int c_thread_x = tid % c_dim_x;

    float Creg[Tm][Tn] = { 0.0f };
    float Areg[Tm] = { 0.0f };
    float Breg[Tn] = { 0.0f };

    for (int k = 0; k < K; k += Bk) {
        // step1 load into shared tile
        #pragma unroll
        for (int i = 0; i < Bm; i += a_dim_y) {
            int row = r0 + i + a_thread_y;
            #pragma unroll
            for (int j = 0; j < Bk; j += a_dim_x) {
                int col = k + j + a_thread_x;
                tile_a[a_thread_x][(i + a_thread_y) ^ (a_thread_x << 1)] = A[row * K + col];
            }
        }
        
        #pragma unroll
        for (int i = 0; i < Bk; i += b_dim_y) {
            int row = k + i + b_thread_y;
            #pragma unroll
            for (int j = 0; j < Bn; j += b_dim_x) {
                int col = c0 + j + b_thread_x;
                tile_b[i + b_thread_y][j + b_thread_x] = B[row * N + col];
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int p = 0; p < Bk; ++p) {
            #pragma unroll
            for (int i = 0; i < Tm; ++i) {
                Areg[i] = tile_a[p][(c_thread_y + i * c_dim_y) ^ (p << 1)];
            }
            
            #pragma unroll
            for (int j = 0; j < Tn; ++j) {
                Breg[j] = tile_b[p][c_thread_x + j * c_dim_x];
            }

            #pragma unroll
            for (int i = 0; i < Tm; ++i) {
                #pragma unroll
                for (int j = 0; j < Tn; ++j) {
                    Creg[i][j] += Areg[i] * Breg[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < Tm; ++i) {
        int row = r0 + c_thread_y + i * c_dim_y;
        for (int j = 0; j < Tn; ++j) {
            int col = c0 + c_thread_x + j * c_dim_x;
            if (row < M && col < N) C[row * N + col] = alpha * Creg[i][j] + beta * C[row * N + col];
        }
    }
}
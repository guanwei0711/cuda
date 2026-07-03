#pragma once
#include <cuda_runtime.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CFLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

template<int Bm = 64, int Bn = 64, int Bk = 8, int Tm = 4, int Tn = 4, int THREADS = 256>
__global__ void v5_gemm_vectorized_access(const float* __restrict__ A, const float* __restrict__ B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ float tile_a[Bk][Bm]; // transposed for vectorized load
    __shared__ float tile_b[Bk][Bn];
    int tid = threadIdx.x;
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;
    
    constexpr int vectorized_size = 4;
    constexpr int a_dim_x = Bk / vectorized_size, a_dim_y = THREADS / a_dim_x;
    constexpr int b_dim_y = Bk, b_dim_x = THREADS / b_dim_y;
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
        for (int i = a_thread_y; i < Bm; i += a_dim_y) {
            int row = r0 + i;
            for (int j = a_thread_x; j < Bk; j += 4 * a_dim_x) {
                int col = k + j * 4;
                float4 tmp = row < M && col < K ? CFLOAT4(A[row * K + col]) : float4(0.0);
                tile_a[j * 4 + 0][i] = tmp.x;
                tile_a[j * 4 + 1][i] = tmp.y;
                tile_a[j * 4 + 2][i] = tmp.z;
                tile_a[j * 4 + 3][i] = tmp.w;
            }
        }

        for (int i = b_thread_y; i < Bk; i += b_dim_y) {
            int row = k + i;
            for (int j = b_thread_x; j < Bn; j += 4 * b_dim_x) {
                int col = c0 + j * 4;
                FLOAT4(tile_b[i][j * 4]) = row < K && col < N ? CFLOAT4(B[row * N + col]) : float4(0.0);
            }
        }
        __syncthreads();
        
        for (int p = 0; p < Bk; ++p) {
            for (int i = 0; i < Tm; ++i) {
                Areg[i] = tile_a[p][c_thread_y * Tm + i];
            }

            for (int j = 0; j < Tn; ++j) {
                Breg[j] = tile_b[p][c_thread_x * Tn + j];
            }

            for (int i = 0; i < Tm; ++i) {
                for (int j = 0; j < Tn; ++j) {
                    Creg[i][j] += Areg[i] * Breg[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < Tm; ++i) {
        int row = r0 + c_thread_y * Tm + i;
        for (int j = 0; j < Tn; ++j) {
            int col = c0 + c_thread_x * Tn + j;
            if (row < M && col < N) C[row * N + col] = alpha * Creg[i][j] + beta * C[row * N + col];
        }
    }
}
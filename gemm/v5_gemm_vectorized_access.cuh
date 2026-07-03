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
        #pragma unroll
        for (int i = 0; i < Bm; i += a_dim_y) {
            int row = r0 + i + a_thread_y;
            #pragma unroll
            for (int j = 0; j < Bk; j += 4 * a_dim_x) {
                int col = k + (j + a_thread_x) * 4;
                float4 tmp = row < M && col < K ? CFLOAT4(A[row * K + col]) : float4{0.0f, 0.0f, 0.0f, 0.0f};;
                tile_a[(j + a_thread_x) * 4 + 0][(i + a_thread_y) ^ ((j + a_thread_x) << 4)] = tmp.x;
                tile_a[(j + a_thread_x) * 4 + 1][(i + a_thread_y) ^ ((j + a_thread_x) << 4)] = tmp.y;
                tile_a[(j + a_thread_x) * 4 + 2][(i + a_thread_y) ^ ((j + a_thread_x) << 4)] = tmp.z;
                tile_a[(j + a_thread_x) * 4 + 3][(i + a_thread_y) ^ ((j + a_thread_x) << 4)] = tmp.w;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < Bk; i += b_dim_y) {
            int row = k + i + b_thread_y;
            #pragma unroll
            for (int j = 0; j < Bn; j += 4 * b_dim_x) {
                int col = c0 + (j + b_thread_x) * 4;
                FLOAT4(tile_b[i + b_thread_y][(j + b_thread_x) * 4]) = row < K && col < N ? CFLOAT4(B[row * N + col]) : float4{0.0f, 0.0f, 0.0f, 0.0f};;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int p = 0; p < Bk; ++p) {
            #pragma unroll
            for (int i = 0; i < Tm / 4; ++i) {
                int col = (c_thread_y + i * c_dim_y) << 2;
                FLOAT4(Areg[i * 4]) = FLOAT4(tile_a[p][col ^ ((p >> 2) << 4)]);
            }

            #pragma unroll
            for (int j = 0; j < Tn / 4; ++j) {
                int col = (c_thread_x + j * c_dim_x) << 2;
                FLOAT4(Breg[j * 4]) = FLOAT4(tile_b[p][col]);
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
        int row = r0 + ((c_thread_y + (i >> 2) * c_dim_y) << 2) + (i % 4);
        for (int j = 0; j < Tn / 4; ++j) {
            int col = c0 + ((c_thread_x + j * c_dim_x) << 2);
            if (row < M && col < N) {
                float4 c = FLOAT4(C[row * N + col]), o;
                o.x = alpha * Creg[i][j * 4 + 0] + beta * c.x;
                o.y = alpha * Creg[i][j * 4 + 1] + beta * c.y;
                o.z = alpha * Creg[i][j * 4 + 2] + beta * c.z;
                o.w = alpha * Creg[i][j * 4 + 3] + beta * c.w;
                FLOAT4(C[row * N + col]) = o;
            }
        }
    }
}
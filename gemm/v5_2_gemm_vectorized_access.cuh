#pragma once
#include <cuda_runtime.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CFLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

template<int Bm = 128, int Bn = 128, int Bk = 8, int Tm = 8, int Tn = 8, int THREADS = 256>
__global__ void v5_2_gemm_vectorized_access(const float* __restrict__ A, const float* __restrict__ B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ float tile_a[2][Bk][Bm]; // transposed for vectorized load
    __shared__ float tile_b[2][Bk][Bn];
    int tile_id = 0;
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
    constexpr int a_smem_load = Bm * Bk / 4 / THREADS;
    constexpr int b_smem_load = Bn * Bk / 4 / THREADS;
    float4 Astage[a_smem_load];
    float4 Bstage[b_smem_load];

    #pragma unroll
    for (int i = 0; i < Bm; i += a_dim_y) {
        int row = r0 + i + a_thread_y;
        int col = a_thread_x * 4;
        int xor_col = (i + a_thread_y) ^ (a_thread_x << 4);
        float4 tmp = CFLOAT4(A[row * K + col]);
        tile_a[tile_id][a_thread_x * 4 + 0][xor_col] = tmp.x;
        tile_a[tile_id][a_thread_x * 4 + 1][xor_col] = tmp.y;
        tile_a[tile_id][a_thread_x * 4 + 2][xor_col] = tmp.z;
        tile_a[tile_id][a_thread_x * 4 + 3][xor_col] = tmp.w;
    }
    
    #pragma unroll
    for (int j = 0; j < Bn; j += 4 * b_dim_x) {
        int row = b_thread_y;
        int col = c0 + j + b_thread_x * 4;
        FLOAT4(tile_b[tile_id][b_thread_y][j + b_thread_x * 4]) = CFLOAT4(B[row * N + col]);
    }
    __syncthreads();

    for (int k = 0; k < K; k += Bk) {
        if (k + Bk < K) {
            int li = 0;
            #pragma unroll
            for (int i = 0; i < Bm; i += a_dim_y) {
                int row = r0 + i + a_thread_y;
                int col = k + a_thread_x * 4;
                Astage[li++] = CFLOAT4(A[row * K + col]);
            }
            
            int lj = 0;
            #pragma unroll
            for (int j = 0; j < Bn; j += 4 * b_dim_x) {
                int row = k + b_thread_y;
                int col = c0 + j + b_thread_x * 4;
                Bstage[lj++] = CFLOAT4(B[row * N + col]);
            }
        }
        
        #pragma unroll
        for (int p = 0; p < Bk; ++p) {
            #pragma unroll
            for (int i = 0; i < Tm / 4; ++i) {
                int col = (c_thread_y + i * c_dim_y) << 2;
                FLOAT4(Areg[i * 4]) = FLOAT4(tile_a[tile_id][p][col ^ ((p >> 2) << 4)]);
            }

            #pragma unroll
            for (int j = 0; j < Tn / 4; ++j) {
                int col = (c_thread_x + j * c_dim_x) << 2;
                FLOAT4(Breg[j * 4]) = FLOAT4(tile_b[tile_id][p][col]);
            }

            if (p == Bk - 1 && k + Bk < K) {
                int li = 0;
                #pragma unroll
                for (int i = 0; i < Bm; i += a_dim_y) {
                    int xor_col = (i + a_thread_y) ^ (a_thread_x << 4);
                    float4 tmp = Astage[li++];
                    tile_a[tile_id ^ 1][a_thread_x * 4 + 0][xor_col] = tmp.x;
                    tile_a[tile_id ^ 1][a_thread_x * 4 + 1][xor_col] = tmp.y;
                    tile_a[tile_id ^ 1][a_thread_x * 4 + 2][xor_col] = tmp.z;
                    tile_a[tile_id ^ 1][a_thread_x * 4 + 3][xor_col] = tmp.w;
                }

                int lj = 0;
                #pragma unroll
                for (int j = 0; j < Bn; j += 4 * b_dim_x) {
                    FLOAT4(tile_b[tile_id ^ 1][b_thread_y][j + b_thread_x * 4]) = Bstage[lj++];
                }
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
        tile_id ^= 1;
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
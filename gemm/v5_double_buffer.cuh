#pragma once
#include <cuda_runtime.h>

template<int Bm = 64, int Bn = 64, int Bk = 8, int Tm = 4, int Tn = 4, int THREADS = 256>
__global__ void v5_double_buffer(const float* __restrict__ A, const float* __restrict__ B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ float tile_a[2][Bm][Bk];
    __shared__ float tile_b[2][Bk][Bn];
    int tile_id = 0;
    int tid = threadIdx.x;
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    constexpr int a_dim_x = Bk, a_dim_y = THREADS / a_dim_x;
    constexpr int b_dim_y = Bk, b_dim_x = THREADS / b_dim_y;
    constexpr int c_dim_x = Bn / Tn, c_dim_y = THREADS / c_dim_x;

    int a_thread_y = tid / a_dim_x;
    int a_thread_x = tid % a_dim_x;

    int b_thread_y = tid / b_dim_x;
    int b_thread_x = tid % b_dim_x;

    int c_thread_y = tid / c_dim_x;
    int c_thread_x = tid % c_dim_x;

    int c_block_y = r0 + c_thread_y * Tm;
    int c_block_x = c0 + c_thread_x;

    float Creg[Tm][Tn] = { 0.0f };
    float Areg[Tm] = { 0.0f };
    float Breg[Tn] = { 0.0f };

    for (int i = a_thread_y; i < Bm; i += a_dim_y) {
        int row = r0 + i;
        for (int j = a_thread_x; j < Bk; j += a_dim_x) {
            int col = j;
            tile_a[tile_id][i][j] = row < M && col < K ? A[row * K + col] : 0.0f;
        }
    }

    for (int i = b_thread_y; i < Bk; i += b_dim_y) {
        int row = i;
        for (int j = b_thread_x; j < Bn; j += b_dim_x) {
            int col = c0 + j;
            tile_b[tile_id][i][j] = row < K && col < N ? B[row * N + col] : 0.0f;
        }
    }

    for (int k = 0; k < K; k += Bk) {
        // step1 load into shared tile
        for (int i = a_thread_y; i < Bm; i += a_dim_y) {
            int row = r0 + i;
            for (int j = a_thread_x; j < Bk; j += a_dim_x) {
                int col = (k + Bk) + j;
                tile_a[tile_id ^ 1][i][j] = row < M && col < K ? A[row * K + col] : 0.0f;
            }
        }

        for (int i = b_thread_y; i < Bk; i += b_dim_y) {
            int row = (k + Bk) + i;
            for (int j = b_thread_x; j < Bn; j += b_dim_x) {
                int col = c0 + j;
                tile_b[tile_id ^ 1][i][j] = row < K && col < N ? B[row * N + col] : 0.0f;
            }
        }
        // __syncthreads();
        
        for (int p = 0; p < Bk; ++p) {
            for (int i = 0; i < Tm; ++i) {
                Areg[i] = tile_a[tile_id][c_thread_y * Tm + i][p];
            }

            for (int j = 0; j < Tn; ++j) {
                Breg[j] = tile_b[tile_id][p][c_thread_x * Tn + j];
            }

            for (int i = 0; i < Tm; ++i) {
                for (int j = 0; j < Tn; ++j) {
                    Creg[i][j] += Areg[i] * Breg[j];
                }
            }
        }

        tile_id ^= 1;
        // __syncthreads();
    }

    for (int i = 0; i < Tm; ++i) {
        int row = r0 + c_thread_y * Tm + i;
        for (int j = 0; j < Tn; ++j) {
            int col = c0 + c_thread_x * Tn + j;
            if (row < M && col < N) C[row * N + col] = alpha * Creg[i][j] + beta * C[row * N + col];
        }
    }
}
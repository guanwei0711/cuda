#pragma once
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void gemm_smem_cached(const float* __restrict__ A, const float* __restrict__ B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ tileB[BLOCK_SIZE][BLOCK_SIZE];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int r0 = blockIdx.y * BLOCK_SIZE;
    int c0 = blockIdx.y * BLOCK_SIZE;
    float sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        int ax = k + x, ay = y + r0;
        int bx = c0 + x, by = k + y;
        tileA[y][x] = (ay < M && ax < K) ? A[ay * K + ax] : 0.0f;
        tileB[y][x] = (by < K && bx < N) ? B[by * N + bx] : 0.0f;
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE) {
            sum += tileA[y][i] * tileB[i][x];
        }
        __syncthreads();
    }

    int col = c0 + x, row = y0 + y;
    if (row < M && col < N) C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
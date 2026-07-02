#pragma once
#include <cuda_runtime.h>

template<int BLOCK_ROW, int BLOCK_COL>
__global__ void v1_gemm_naive(const float* __restrict__ A, const float* __restrict__ B, float *C, int M, int K, int N, float alpha, float beta) {
    int y = BLOCK_ROW * blockIdx.y + threadIdx.y;
    int x = BLOCK_COL * blockIdx.x + threadIdx.x;

    if (y < M && x < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[y * K + i] * B[i * N + x];
        }
        C[y * N + x] = alpha * sum + beta * C[y * N + x];
    }
}
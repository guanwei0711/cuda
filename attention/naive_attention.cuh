#pragma once
#include <cuda_runtime.h>

__global__ void naive_attention(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    float* __restrict__ Sscratch,          // N*N global scratch for scores
    int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // query row
    if (i >= N) return;
    float* s = &Sscratch[(size_t)i * N];             // this row's scores

    // 1) scores + running max
    float m = -FLT_MAX;
    for (int j = 0; j < N; ++j) {
        float dot = 0.f;
        for (int k = 0; k < d; ++k) dot += Q[i*d+k] * K[j*d+k];
        s[j] = dot;
        m = fmaxf(m, dot);
    }
    // 2) exp + sum
    float l = 0.f;
    for (int j = 0; j < N; ++j) { s[j] = __expf(s[j] - m); l += s[j]; }
    // 3) weighted sum of V, normalized
    for (int k = 0; k < d; ++k) {
        float acc = 0.f;
        for (int j = 0; j < N; ++j) acc += s[j] * V[j*d+k];
        O[i*d+k] = acc / l;
    }
}
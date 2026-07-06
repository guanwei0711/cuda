#pragma once
#include <cuda_runtime.h>
template<int Br = 32, int Bc = 32, int STRIDE = 32>
__global__ void flash_attention_v2(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ output, int N, int d)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int r0   = blockIdx.y * Br;

    extern __shared__ float smem[];
    float* q_tile  = smem;                       // Br * d
    float* kv_tile = &q_tile[Br * d];            // Bc * d  (reused for K then V)
    float* s_tile  = &kv_tile[Bc * d];           // Br * Bc

    float row_sum = 0.0f;
    float row_max = -FLT_MAX;

    float o_acc[ (32 + 31) / 32 * 32 ]; 
    for (int i = 0; i < d; i += STRIDE) o_acc[i/STRIDE] = 0.0f;

    for (int i = 0; i < d; i += STRIDE)
        q_tile[tidy * d + i + tidx] = Q[(r0 + tidy) * d + i + tidx];

    for (int c0 = 0; c0 < N; c0 += Bc) {
        for (int i = 0; i < d; i += STRIDE)
            kv_tile[tidy * d + i + tidx] = K[(c0 + tidy) * d + i + tidx];
        __syncthreads();

        float sij = 0.0f;
        for (int i = 0; i < d; i++)
            sij += q_tile[tidy * d + i] * kv_tile[tidx * d + i];

        float mblk = sij;
        #pragma unroll
        for (int off = Bc/2; off; off >>= 1)
            mblk = fmaxf(mblk, __shfl_down_sync(0xffffffff, mblk, off));
        mblk = __shfl_sync(0xffffffff, mblk, 0);   // broadcast lane 0

        float mnew       = fmaxf(row_max, mblk);
        float correction = __expf(row_max - mnew);

        float pij = __expf(sij - mnew);
        s_tile[tidy * Bc + tidx] = pij;

        float lblk = pij;
        #pragma unroll
        for (int off = Bc/2; off; off >>= 1)
            lblk += __shfl_down_sync(0xffffffff, lblk, off);
        lblk = __shfl_sync(0xffffffff, lblk, 0);

        row_sum = correction * row_sum + lblk;
        __syncthreads();

        for (int i = 0; i < d; i += STRIDE)
            kv_tile[tidy * d + i + tidx] = V[(c0 + tidy) * d + i + tidx];
        __syncthreads();

        for (int i = 0; i < d; i += STRIDE) {
            float sum = 0.0f;
            for (int j = 0; j < Bc; ++j)
                sum += s_tile[tidy * Bc + j] * kv_tile[j * d + i + tidx];
            o_acc[i/STRIDE] = correction * o_acc[i/STRIDE] + sum;
        }
        __syncthreads();

        row_max = mnew;
    }

    for (int i = 0; i < d; i += STRIDE)
        output[(r0 + tidy) * d + i + tidx] = o_acc[i/STRIDE] / row_sum;
}
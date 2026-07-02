#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include "v1_gemm_naive.cuh"
#include "v2_gemm_smem_cached.cuh"
#include "v3_gemm_1d_tiling.cuh"
#include "v4_gemm_2d_tiling.cuh"

void gemm_cpu(const std::vector<float>& A,const std::vector<float>& B,std::vector<float>& C, int M, int K, int N, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main() {
    int DIM = 256;
    int M = DIM, K = DIM, N = DIM;
    size_t sizeA = M * K, sizeB = K * N, sizeC = M * N;
    std::vector<float> hA, hB, hC, hC_cpu, hC_kernel;
    hA.resize(sizeA);
    hB.resize(sizeB);
    hC.resize(sizeC);
    hC_kernel.resize(sizeC);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v: hA) v = dist(rng);
    for (auto& v: hB) v = dist(rng);
    for (auto& v: hC) v = dist(rng);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(float) * sizeA);
    cudaMalloc(&dB, sizeof(float) * sizeB);
    cudaMalloc(&dC, sizeof(float) * sizeC);

    cudaMemcpy(dA, hA.data(), sizeof(float) * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(float) * sizeB, cudaMemcpyHostToDevice);

    float alpha = dist(rng);
    float beta = dist(rng);

    bool check_correctness = (M <= 1024 && K <= 1024 && N <= 1024);
    if (check_correctness) {
        hC_cpu = hC;
        printf("Running CPU reference...\n");
        gemm_cpu(hA, hB, hC_cpu, M, K, N, alpha, beta);
    } else {
        printf("Skipping CPU reference...\n");
    }

    {
        constexpr int BLOCK_SIZE = 16;
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        gemm_naive<<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("Naive kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr int BLOCK_SIZE = 16;
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        gemm_smem_cached<<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("Block tiled kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr Bm = 64, Bn = 64;
        constexpr Bk = 8, Tm = 16;
        constexpr THREADS = 256;
        dim3 threads(THREADS);
        dim3 blocks(N, (M + Tm - 1) / Tm);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        v3_gemm_1d_tiling<Bm, Bn, Bk, Tm, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("Block tiled kernel max relative error: %e\n", err);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
#include <vector>
#include <cstdio>
#include <random>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include "v1_gemm_naive.cuh"
#include "v2_gemm_smem_cached.cuh"
#include "v3_gemm_1d_tiling.cuh"
#include "v4_gemm_2d_tiling.cuh"
#include "v5_global_store_coalesced.cuh"
#include "v5_gemm_vectorized_access.cuh"
#include "v6_elude_bank_conflict.cuh"

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

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = a.size();
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (!std::isfinite(diff)) return INFINITY;
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

int main(int argc, char** argv) {
    int DIM = (argc > 1) ? std::atoi(argv[1]) : 2048;
    if (DIM <= 0) {
        std::fprintf(stderr, "invalid DIM '%s'; using 2048\n", argv[1]);
        DIM = 2048;
    }

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
        v1_gemm_naive<BLOCK_SIZE, BLOCK_SIZE><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("naive kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr int BLOCK_SIZE = 16;
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        v2_gemm_smem_cached<BLOCK_SIZE><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("smem cached kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr int Bm = 64, Bn = 64;
        constexpr int Bk = 8, Tm = 16;
        constexpr int THREADS = 256;
        dim3 threads(THREADS);
        dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        v3_gemm_1d_tiling<Bm, Bn, Bk, Tm, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("1d reg tiling kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr int Bm = 64, Bn = 64, Bk = 16;
        constexpr int Tm = 4, Tn = 4;
        constexpr int THREADS = 256;
        dim3 threads(THREADS);
        dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        v4_gemm_2d_tiling<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("2d reg tiling Bk=16 kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr int Bm = 64, Bn = 64, Bk = 16;
        constexpr int Tm = 4, Tn = 4;
        constexpr int THREADS = 256;
        dim3 threads(THREADS);
        dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        v5_gemm_vectorized_access<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("memory vectorized (64, 64, 16) kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr int Bm = 128, Bn = 128, Bk = 16;
        constexpr int Tm = 8, Tn = 8;
        constexpr int THREADS = 256;
        dim3 threads(THREADS);
        dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        v5_gemm_vectorized_access<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) printf("cfg fail: %s (grid %d,%d block %d)\n",
            cudaGetErrorString(e), blocks.x, blocks.y, threads.x);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("memory vectorized (128, 128, 16) kernel max relative error: %e\n", err);
        }
    }

    {
        constexpr int Bm = 128, Bn = 128, Bk = 8;
        constexpr int Tm = 8, Tn = 8;
        constexpr int THREADS = 256;
        dim3 threads(THREADS);
        dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
        cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
        v5_gemm_vectorized_access<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        if (check_correctness) {
            cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
            float err = max_abs_error(hC_cpu, hC_kernel);
            printf("memory vectorized (128, 128, 8) kernel max relative error: %e\n", err);
        }
    }

    // {
    //     constexpr int Bm = 64, Bn = 64, Bk = 16;
    //     constexpr int Tm = 4, Tn = 4;
    //     constexpr int THREADS = 256;
    //     dim3 threads(THREADS);
    //     dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
    //     cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
    //     v6_elude_bank_conflict<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
    //     cudaDeviceSynchronize();
    //     if (check_correctness) {
    //         cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
    //         float err = max_abs_error(hC_cpu, hC_kernel);
    //         printf("elude bank conflict Bk=16 kernel max relative error: %e\n", err);
    //     }
    // }

    // {
    //     constexpr int Bm = 32, Bn = 32, Bk = 8;
    //     constexpr int Tm = 2, Tn = 2;
    //     constexpr int THREADS = 256;
    //     dim3 threads(THREADS);
    //     dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
    //     cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
    //     v4_gemm_2d_tiling<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
    //     cudaDeviceSynchronize();
    //     if (check_correctness) {
    //         cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
    //         float err = max_abs_error(hC_cpu, hC_kernel);
    //         printf("2d reg tiling (2x2) kernel max relative error: %e\n", err);
    //     }
    // }

    // {
    //     constexpr int Bm = 128, Bn = 128, Bk = 8;
    //     constexpr int Tm = 8, Tn = 8;
    //     constexpr int THREADS = 256;
    //     dim3 threads(THREADS);
    //     dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
    //     cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
    //     v4_gemm_2d_tiling<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
    //     cudaDeviceSynchronize();
    //     if (check_correctness) {
    //         cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
    //         float err = max_abs_error(hC_cpu, hC_kernel);
    //         printf("2d reg tiling (8x8) kernel max relative error: %e\n", err);
    //     }
    // }

    // {
    //     constexpr int Bm = 64, Bn = 64, Bk = 8;
    //     constexpr int Tm = 4, Tn = 4;
    //     constexpr int THREADS = 256;
    //     dim3 threads(THREADS);
    //     dim3 blocks((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
    //     cudaMemcpy(dC, hC.data(), sizeof(float) * sizeC, cudaMemcpyHostToDevice);
    //     v5_global_store_coalesced<Bm, Bn, Bk, Tm, Tn, THREADS><<<blocks, threads>>>(dA, dB, dC, M, K, N, alpha, beta);
    //     cudaDeviceSynchronize();
    //     if (check_correctness) {
    //         cudaMemcpy(hC_kernel.data(), dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
    //         float err = max_abs_error(hC_cpu, hC_kernel);
    //         printf("global store coalesced kernel max relative error: %e\n", err);
    //     }
    // }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
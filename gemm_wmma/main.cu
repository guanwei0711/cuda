#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "v1_gemm_naive.cu"
#include "v2_gemm_smem_tiled.cu"

void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
              int M, int K, int N, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (!std::isfinite(diff)) return INFINITY;
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

int main() {
    int M = 2048, K = 2048, N = 2048;   // multiple of 16 (v1 tile) and 32 (v2 tile): no edge-padding needed
    size_t sizeA = (size_t)M * K, sizeB = (size_t)K * N, sizeC = (size_t)M * N;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<half> hA(sizeA), hB(sizeB), hC(sizeC);
    std::vector<float> hA_ref(sizeA), hB_ref(sizeB), hC_ref(sizeC);

    for (size_t i = 0; i < sizeA; ++i) { hA[i] = __float2half(dist(rng)); hA_ref[i] = __half2float(hA[i]); }
    for (size_t i = 0; i < sizeB; ++i) { hB[i] = __float2half(dist(rng)); hB_ref[i] = __half2float(hB[i]); }
    for (size_t i = 0; i < sizeC; ++i) { hC[i] = __float2half(dist(rng)); hC_ref[i] = __half2float(hC[i]); }

    float alpha = dist(rng), beta = dist(rng);

    printf("Running CPU reference...\n");
    std::vector<float> hC_cpu = hC_ref;
    gemm_cpu(hA_ref, hB_ref, hC_cpu, M, K, N, alpha, beta);

    half *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(half) * sizeA);
    cudaMalloc(&dB, sizeof(half) * sizeB);
    cudaMalloc(&dC, sizeof(half) * sizeC);
    cudaMemcpy(dA, hA.data(), sizeof(half) * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(half) * sizeB, cudaMemcpyHostToDevice);

    std::vector<half> hC_out(sizeC);
    std::vector<float> hC_out_f(sizeC);
    auto to_float = [&](const std::vector<half>& h, std::vector<float>& f) {
        for (size_t i = 0; i < h.size(); ++i) f[i] = __half2float(h[i]);
    };

    {
        dim3 threads(32, 1);
        dim3 blocks(N / 16, M / 16);
        cudaMemcpy(dC, hC.data(), sizeof(half) * sizeC, cudaMemcpyHostToDevice);
        v1_gemm_naive<<<blocks, threads>>>(dA, dB, dC, M, N, K, alpha, beta);
        cudaDeviceSynchronize();
        cudaMemcpy(hC_out.data(), dC, sizeof(half) * sizeC, cudaMemcpyDeviceToHost);
        to_float(hC_out, hC_out_f);
        printf("v1_gemm_naive      max abs error: %e\n", max_abs_error(hC_cpu, hC_out_f));
    }

    {
        dim3 threads(WARP_SIZE * WARPS, 1);
        dim3 blocks((N + N_SMEM_COLS - 1) / N_SMEM_COLS, (M + M_SMEM_ROWS - 1) / M_SMEM_ROWS);
        cudaMemcpy(dC, hC.data(), sizeof(half) * sizeC, cudaMemcpyHostToDevice);
        v2_gemm_smem_tiled<<<blocks, threads>>>(dA, dB, dC, M, N, K, alpha, beta);
        cudaDeviceSynchronize();
        cudaMemcpy(hC_out.data(), dC, sizeof(half) * sizeC, cudaMemcpyDeviceToHost);
        to_float(hC_out, hC_out_f);
        printf("v2_gemm_smem_tiled max abs error: %e\n", max_abs_error(hC_cpu, hC_out_f));
    }

    {
        cublasHandle_t handle = nullptr;
        cublasCreate(&handle);
        cudaMemcpy(dC, hC.data(), sizeof(half) * sizeC, cudaMemcpyHostToDevice);
        // row-major C = A*B via cuBLAS's column-major API: swap operand order and (M,N).
        // CUBLAS_GEMM_DEFAULT_TENSOR_OP routes fp16 inputs through the tensor core (WMMA) path.
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                     &beta, dC, CUDA_R_16F, N,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        cudaMemcpy(hC_out.data(), dC, sizeof(half) * sizeC, cudaMemcpyDeviceToHost);
        to_float(hC_out, hC_out_f);
        printf("cublas wmma gemm   max abs error: %e\n", max_abs_error(hC_cpu, hC_out_f));
        cublasDestroy(handle);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

__global__ void v1_gemm_naive(const half* A, const half* B, half* C,
                               int M, int N, int K, float alpha, float beta) {
    int warp_m = blockIdx.y;
    int warp_n = blockIdx.x;
    int row = warp_m * 16;
    int col = warp_n * 16;
    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;   // math in fp32
    wmma::fragment<wmma::accumulator, 16, 16, 16, half>  c_frag;     // matches C's dtype

    wmma::fill_fragment(acc_frag, 0.f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::load_matrix_sync(c_frag, C + row * N + col, N, wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; ++i)
        c_frag.x[i] = __float2half(alpha * acc_frag.x[i] + beta * __half2float(c_frag.x[i]));
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

// additional padding

// static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// #define CUDA_CHECK(call)                                                       \
//     do {                                                                       \
//         cudaError_t err_ = (call);                                             \
//         if (err_ != cudaSuccess) {                                             \
//             fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
//                     cudaGetErrorString(err_), __FILE__, __LINE__);             \
//             return;                                                            \
//         }                                                                      \
//     } while (0)

// // A, B, and C are device pointers
// extern "C" void solve(const half* A, const half* B, half* C,
//                       int M, int N, int K, float alpha, float beta) {
//     const int Mp = round_up(M, 16);
//     const int Np = round_up(N, 16);
//     const int Kp = round_up(K, 16);

//     // Fast path: already aligned, no padding needed.
//     if (Mp == M && Np == N && Kp == K) {
//         dim3 threads(32, 1);
//         dim3 blocks(N / 16, M / 16);
//         gemm_fp16_wmma<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
//         CUDA_CHECK(cudaGetLastError());
//         CUDA_CHECK(cudaDeviceSynchronize());
//         return;
//     }

//     half *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
//     CUDA_CHECK(cudaMalloc(&Ap, (size_t)Mp * Kp * sizeof(half)));
//     CUDA_CHECK(cudaMalloc(&Bp, (size_t)Kp * Np * sizeof(half)));
//     CUDA_CHECK(cudaMalloc(&Cp, (size_t)Mp * Np * sizeof(half)));

//     // Zero all three: padded rows/cols of A and B must contribute 0 to the dot
//     // product, and Cp's padded region is read by the beta*C epilogue.
//     CUDA_CHECK(cudaMemset(Ap, 0, (size_t)Mp * Kp * sizeof(half)));
//     CUDA_CHECK(cudaMemset(Bp, 0, (size_t)Kp * Np * sizeof(half)));
//     CUDA_CHECK(cudaMemset(Cp, 0, (size_t)Mp * Np * sizeof(half)));

//     // Copy the real data into the top-left corner of each padded buffer.
//     // pitch = row stride in BYTES; width = bytes per row; height = rows.
//     CUDA_CHECK(cudaMemcpy2D(Ap, (size_t)Kp * sizeof(half),
//                             A,  (size_t)K  * sizeof(half),
//                             (size_t)K * sizeof(half), M,
//                             cudaMemcpyDeviceToDevice));

//     CUDA_CHECK(cudaMemcpy2D(Bp, (size_t)Np * sizeof(half),
//                             B,  (size_t)N  * sizeof(half),
//                             (size_t)N * sizeof(half), K,
//                             cudaMemcpyDeviceToDevice));

//     if (beta != 0.0f) {
//         CUDA_CHECK(cudaMemcpy2D(Cp, (size_t)Np * sizeof(half),
//                                 C,  (size_t)N  * sizeof(half),
//                                 (size_t)N * sizeof(half), M,
//                                 cudaMemcpyDeviceToDevice));
//     }

//     dim3 threads(32, 1);
//     dim3 blocks(Np / 16, Mp / 16);
//     gemm_fp16_wmma<<<blocks, threads>>>(Ap, Bp, Cp, Mp, Np, Kp, alpha, beta);
//     CUDA_CHECK(cudaGetLastError());

//     // Copy the valid M×N region back out.
//     CUDA_CHECK(cudaMemcpy2D(C,  (size_t)N  * sizeof(half),
//                             Cp, (size_t)Np * sizeof(half),
//                             (size_t)N * sizeof(half), M,
//                             cudaMemcpyDeviceToDevice));

//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaFree(Ap));
//     CUDA_CHECK(cudaFree(Bp));
//     CUDA_CHECK(cudaFree(Cp));
// }
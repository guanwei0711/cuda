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
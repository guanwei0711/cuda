#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include "naive_attention.cuh"
#include "flash_attention_v2.cuh"

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// --------------------------- CPU reference --------------------------------
static void attention_ref(const float* Q, const float* K, const float* V,
                          float* out, int N, int d) {
    float* s = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        float m = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            float dot = 0.f;
            for (int k = 0; k < d; ++k) dot += Q[i*d+k]*K[j*d+k];
            s[j] = dot; m = fmaxf(m, dot);
        }
        float l = 0.f;
        for (int j = 0; j < N; ++j) { s[j] = expf(s[j]-m); l += s[j]; }
        for (int k = 0; k < d; ++k) {
            float acc = 0.f;
            for (int j = 0; j < N; ++j) acc += s[j]*V[j*d+k];
            out[i*d+k] = acc / l;
        }
    }
    free(s);
}

float max_abs_error(const float* hRef, const float* hGpu, int N, int d) {
    float maxerr = -FLT_MAX;
    for (int i=0;i< N*d ;i++){ 
        double e=fabs(hRef[i]-hGpu[i]); 
        if(e>maxerr){ maxerr=e; } 
    }
    return maxerr;
}

int main() {
    const int N = 64, d = 32;

    size_t sz = (size_t)N*d*sizeof(float);
    float *hQ=(float*)malloc(sz),*hK=(float*)malloc(sz),*hV=(float*)malloc(sz);
    float *hRef=(float*)malloc(sz),*hGpu=(float*)malloc(sz);
    srand(42);
    for (int i=0;i<N*d;i++){ hQ[i]=rand()/(float)RAND_MAX-0.5f;
                             hK[i]=rand()/(float)RAND_MAX-0.5f;
                             hV[i]=rand()/(float)RAND_MAX-0.5f; }

    attention_ref(hQ,hK,hV,hRef,N,d);

    float *dQ,*dK,*dV,*dO,*dS;
    CUDA_CHECK(cudaMalloc(&dQ,sz)); CUDA_CHECK(cudaMalloc(&dK,sz));
    CUDA_CHECK(cudaMalloc(&dV,sz)); CUDA_CHECK(cudaMalloc(&dO,sz));
    CUDA_CHECK(cudaMalloc(&dS,(size_t)N*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,sz,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,hK,sz,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,hV,sz,cudaMemcpyHostToDevice));

    {
        constexpr int threads = 128, blocks = (N + threads - 1) / threads;
        naive_attention<<<blocks,threads>>>(dQ,dK,dV,dO,dS,N,d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hGpu,dO,sz,cudaMemcpyDeviceToHost));
        float err = max_abs_error(hRef, hGpu, N, d);
        printf("naive attention kernel max relative error: %e\n", err);
    }

    {
        const int Br = 32, Bc = 32, STRIDE = 32;
        dim3 block(Bc, Br);
        dim3 grid(1, (N + Br - 1) / Br);
        size_t shmem = (Br*d + Bc*d + Br*Bc) * sizeof(float);

        flash_attention_v2<Br,Bc,STRIDE><<<grid,block,shmem>>>(dQ,dK,dV,dO,N,d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hGpu,dO,sz,cudaMemcpyDeviceToHost));
        float err = max_abs_error(hRef, hGpu, N, d);
        printf("v2 flast attention kernel max relative error: %e\n", err);
    }

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dS);
    free(hQ);free(hK);free(hV);free(hRef);free(hGpu);
    return 0;
}

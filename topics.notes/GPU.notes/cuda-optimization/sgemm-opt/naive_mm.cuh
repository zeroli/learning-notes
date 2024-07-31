#ifndef NAIVE_GEMM_CUH_
#define NAIVE_GEMM_CUH_

#include "common.h"

namespace sgemm_kernel {
__global__ void smem_naive(const float* A, const float* B, float* C,
                                    int M, int N, int K)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < M && tx < N) {
        float c = 0.f;
        for (int k = 0; k < K; k++) {
            c += A[OFFSET(ty, k, K)] * B[OFFSET(k, tx, N)];
        }
        C[OFFSET(ty, tx, N)] = c;
    }
}
void run_kernel_naive(const float* A, const float* B, float* C,
                                    int M, int N, int K)
{
    constexpr int size = 32;
    // block size = 32 x 32, each thread computes 1 result
    dim3 blockSz(size, size);
    dim3 gridSz((N + size - 1) / size, (M + size - 1) / size);
    smem_naive<<<gridSz, blockSz>>>(A, B, C, M, N, K);
}
}  // namespace sgemm_kernel

#endif  // NAIVE_GEMM_CUH_
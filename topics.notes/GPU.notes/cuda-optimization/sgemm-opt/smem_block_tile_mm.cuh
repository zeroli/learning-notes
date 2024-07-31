#ifndef SMEM_1_MM_CUH_
#define SMEM_1_MM_CUH_

#include "common.h"

namespace sgemm_kernel {
template <int BM, int BN, int BK>
__global__ void smem_block_tile(const float* A, const float* B, float* C,
                                    int M, int N, int K)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty >= M || tx >= N) {
        return;
    }
    __shared__ float As[BM][BK]; // BM x BK
    __shared__ float Bs[BK][BN];  // BK x BN

    int local_tx = threadIdx.x;
    int local_ty = threadIdx.y;

    // move A/B pointers to block local starting pointers
    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    // no bank conflict for shared memory of As/Bs
    // since a warp (32 threads) access As one element at one read
    // access Bs 32 floats at one read from 32 threads, one row
    float c = 0.f;
    for (int i = 0; i < K; i += BK)
    {
        // load data from global memory to shared memory
        __syncthreads();
        As[local_ty][local_tx] = A[OFFSET(local_ty, local_tx, K)];
        Bs[local_ty][local_tx] = B[OFFSET(local_ty, local_tx, N)];
        __syncthreads();

        // move to next block tile
        A += BK;
        B += BK * N;

        {  // one thread computes 1 x BK * BK x 1
            for (int k = 0; k < BK; k++) {
                c += As[local_ty][k] * Bs[k][local_tx];
            }
        }
    }
    C[OFFSET(local_ty, local_tx, N)] = c;
}

void run_kernel_block_tile(const float* A, const float* B, float* C,
                                    int M, int N, int K)
{
    constexpr int size = 32;
    constexpr int BM = size;
    constexpr int BN = size;
    constexpr int BK = size;
    // block size = 32 x 32, each thread computes 1 result
    dim3 blockSz(size, size);
    dim3 gridSz((N + size - 1) / size, (M + size - 1) / size);
    smem_block_tile<BM, BN, BK><<<gridSz, blockSz>>>(A, B, C, M, N, K);
}
}  // namespace sgemm_kernel

#endif  // SMEM_1_MM_CUH_
#ifndef SMEM_2d_tile_MM_CUH_
#define SMEM_2d_tile_MM_CUH_

#include "common.h"

namespace sgemm_kernel {
template <int BM, int BN, int BK, int TM, int TN>
__global__ void smem_2d_tile(const float* A, const float* B, float* C,
                                    int M, int N, int K)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty * TM >= M || tx * TN >= N) {
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

    // each thread will compute TM * 1 result of C
    float c[TM][TN] = {{0.f}}; // 4x4 for one thread
    for (int i = 0; i < K; i += BK)
    {
        // load data from global memory to shared memory
        __syncthreads();
        #pragma unroll
        for (int ik = 0; ik < TM; ik++) {
            int row = local_ty * TM + ik;
            int col = local_tx;
            As[row][col] = A[OFFSET(row, col, K)];
        }
        #pragma unroll
        for (int ik = 0; ik < TN; ik++) {
            int row = local_ty;
            int col = local_tx * TN + ik;
            Bs[row][col] = B[OFFSET(row, col, N)];
        }
        __syncthreads();

        // move to next block tile
        A += BK;
        B += BK * N;

        {  // one thread computes
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                // compute 4x4 partial result from 4x1 @ 1x4
                #pragma unroll
                for (int im = 0; im < TM; im++) {
                    #pragma unroll
                    for (int in = 0; in < TN; in++) {
                        c[im][in] += As[local_ty * TM + im][k] * Bs[k][local_tx * TN + in];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int im = 0; im < TM; im++) {
        #pragma unroll
        for (int in = 0; in < TN; in++) {
            C[OFFSET(local_ty * TM + im, local_tx * TN + in, N)] = c[im][in];
        }
    }
}

void run_kernel_2d_tile(const float* A, const float* B, float* C,
                                    int M, int N, int K)
{
    constexpr int size = 32;  // use 32 instead of 16 to avoid bank conflict 2-way
    constexpr int tile_size = 4;
    constexpr int BM = size * tile_size;  // 32 * 4
    constexpr int BN = size * tile_size;  //  32 * 4
    constexpr int BK = size;  // 32
    // 128 x 32  @ 32 x 128
    // block size = 32 x 32, each thread computes 4x4 results
    dim3 blockSz(size, size);
    dim3 gridSz((N + size - 1) / size, (M + size - 1) / size);
    smem_2d_tile<BM, BN, BK, tile_size, tile_size><<<gridSz, blockSz>>>(A, B, C, M, N, K);
}
}  // namespace sgemm_kernel

#endif  // SMEM_2d_tile_MM_CUH_
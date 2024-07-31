#ifndef CPU_MM_H
#define CPU_MM_H

#include "common.h"

namespace sgemm_kernel {
void cpu(const float* A, const float* B, float* C,
               int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[OFFSET(i, k, K)] * B[OFFSET(k, j, N)];
            }
            C[OFFSET(i, j, N)] = sum;
        }
    }
}
}  // namespace sgemm_kernel

#endif  // CPU_MM_H
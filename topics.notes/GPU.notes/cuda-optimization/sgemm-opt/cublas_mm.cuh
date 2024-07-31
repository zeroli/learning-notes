#ifndef CUBLAS_MM_CUH_
#define CUBLAS_MM_CUH_

#include "common.h"

#include <cublas_v2.h>

namespace sgemm_kernel {
struct Cublas {
    Cublas() {
        cublasCreate(&cublas_handle);
    }
    ~Cublas() {
        cublasDestroy(cublas_handle);
    }
    cublasHandle_t cublas_handle;
};

static Cublas cublas_hdl;

void run_kernel_cublas(const float* A, const float* B, float* C,
                                    int M, int N, int K)
{
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    cublasSgemm(cublas_hdl.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &cublas_alpha,
        B, N,
        A, K,
        &cublas_beta,
        C,
        N
    );
}
}  // namespace sgemm_kernel

#endif  // CUBLAS_MM_CUH_
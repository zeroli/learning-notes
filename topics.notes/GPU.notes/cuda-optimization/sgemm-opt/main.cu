#include <iostream>
#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sys/time.h>

#include "common.h"

#include "cublas_mm.cuh"
#include "naive_mm.cuh"
#include "smem_block_tile_mm.cuh"
#include "smem_2d_tile_mm.cuh"
#include "smem_2d_tile_float4_mm.cuh"

using namespace std;

#define CHECK_CUDA(call) \
    assert((call) == cudaSuccess)

enum OPT_KERNEL {
    cublas = 0,
    naive = 1,
    smem_block_tile = 2,
    smem_2d_tile = 3,
    smem_2d_tile_float4 = 4,
};

typedef void (*gpuSgemm_t) (const float *, const float *, float *, int, int, int);

gpuSgemm_t get_opt_kernel(int opt_kernel)
{
    switch (opt_kernel) {
        case OPT_KERNEL::cublas:
        {
            fprintf(stderr, "kernel:  cublas\n");
            return sgemm_kernel::run_kernel_cublas;
            break;
        }
        case OPT_KERNEL::naive:
        {
            fprintf(stderr, "kernel:  naive\n");
            return sgemm_kernel::run_kernel_naive;
            break;
        }
        case OPT_KERNEL::smem_block_tile:
        {
            fprintf(stderr, "kernel:  smem_block_tile\n");
            return sgemm_kernel::run_kernel_block_tile;
            break;
        }
        case OPT_KERNEL::smem_2d_tile:
        {
            fprintf(stderr, "kernel:  smem_2d_tile\n");
            return sgemm_kernel::run_kernel_2d_tile;
            break;
        }
        case OPT_KERNEL::smem_2d_tile_float4:
        {
            fprintf(stderr, "kernel:  smem_2d_tile_float4\n");
            return sgemm_kernel::run_kernel_2d_tile_float4;
            break;
        }
        default:
        {
            return nullptr;
            break;
        }
    }
}

float test(gpuSgemm_t gpuSgemm, const int M, const int N, const int K, const int repeat)
{
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size_a));
    CHECK_CUDA(cudaMalloc(&d_b, size_b));
    CHECK_CUDA(cudaMalloc(&d_c, size_c));

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        gpuSgemm(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    float msec, sec;
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, end));
    sec = msec / 1000.0 / repeat;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return sec;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: sgemm <opt_kernel>\n");
        exit(-1);
    }

    int opt_kernel = atoi(argv[1]);
    fprintf(stderr, "sgem kernel: %d\n", opt_kernel);
    const int M_list[] = { 1024 };
    const int N_list[] = { 1024 };
    const int K_list[] = { 1024 };

    const int outer_repeat = 10, inner_repeat = 1;
    const int TESTNUM = sizeof(M_list) / sizeof(int);

    gpuSgemm_t gpuSgemm = get_opt_kernel(opt_kernel);

    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = test(gpuSgemm, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf s, AVG Performance = %10.4lf Gflops\n",
            M, N, K, avg_sec, avg_Gflops);
    }
}
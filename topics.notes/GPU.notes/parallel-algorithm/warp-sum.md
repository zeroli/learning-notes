```c++
#include <iostream>
#include <cassert>
#include <cstdio>

#define CHECK(call) do { \
    auto ret = (call); \
    if (ret != cudaSuccess) { \
        printf("call %s failed: %s\n", #call, cudaGetErrorString(ret)); \
        assert(0); \
    } \
} while (0)

__device__ int WarpSum(int val)
{
    for (int diff = warpSize / 2; diff > 0; diff >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, diff);
    }   
    return val;
}

__global__ void SumKernel(int* data, int n, int* sum)
{
    __shared__ int shared[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int my_lane = idx % warpSize;
    int my_warp = idx / warpSize;
    if (idx < n) {
        int val = data[idx];
        int warpSum = WarpSum(val);
        if (my_lane == 0) {  // don't need other "partial" warpSum from other threads in same warp
            shared[my_warp] = warpSum;
        }
        __syncthreads();

        if (my_warp == 0) {  // first warp ?? 
            val = shared[my_lane];  // copy to one lane of warp 0
            int blockSum = WarpSum(val);
            if (idx == 0) {  // warp 0, thread 0 (first thread in block)
                atomicAdd(sum, blockSum);
            }
        }
    }   
}

int main()
{
    int n = 1024;
    int* data = nullptr;
    CHECK(cudaMallocManaged(&data, sizeof(int) * n));
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }   

    dim3 blocks(32 * 32);
    dim3 grids((n + blocks.x - 1) / blocks.x);
    int* sum = nullptr;
    CHECK(cudaMallocManaged(&sum, sizeof(int)));
    SumKernel<<<grids, blocks>>>(data, n, sum);
    cudaDeviceSynchronize();

    std::cout << "sum: " << *sum << "\n";
    CHECK(cudaFree(data));
}
```

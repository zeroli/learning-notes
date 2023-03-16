# Date: 03/09/2023
# 开源代码学习计划和进度跟踪
----------------------------------------------------------

| 项目名字 | 范畴 | 实现语言 | 进度 | comment
| --- | --- | --- | --- | --- |
| thrust | 并行计算算法库 | C++/CUDA/tbb | 正阅读 cuda实现部分 | 针对tbb/openmp/cuda有不同的实现代码，外围接口一致，cuda采用CUB库来实现 |
| AMD HIP API  | GPU driver/runtime 实现 | C++ | | |
| AMD HIP OpenCL runtime | AMD GPU OpenCL实现 | C++ | | |
| oneDNN-0.1 | deep learning framework | C++/xbyak-JIT | 已阅读到xbyak实现部分，手动生成asm代码，然后JIT编译成BINARY执行。这部分需要等到看完那边汇编的书再来研究 |


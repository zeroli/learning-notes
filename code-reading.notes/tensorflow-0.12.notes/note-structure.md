# TF-0.12的代码目录结构的学习

- tensorflow
  - core
    - common_runtime: session, executor, device就定义在这个目录
    - debug
    - distributed_runtime: 分布式计算的运行时
    - framework: 框架层
    - graph: 计算图
    - kernels: 计算的kernel
    - lib: 一些基础库代码
    - ops: 计算图里对应的operation
    - platform: 平台代码，这个目录代码比较多
    - protobuf: 一些配置选项对应的proto文件
  - stream_executor

TF创建一个Session，然后attach一个graph进行运算，在graph计算过程中，会有多个线程同时运行
- 问题：为啥会有多个线程呢?
- 问题：每个节点的运算时如何dispatch到device上进行运算的呢？
- 问题：TF如何解析graph，从而获取node的信息，进行怎么样的运算？
- 问题：如果有一些节点运算时并行的，那么TF是如何做到下一个依赖的节点等待被依赖节点运算结束的呢？

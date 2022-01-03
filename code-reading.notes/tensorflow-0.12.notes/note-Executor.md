# Executor

一段关于Executor的用法示例：
```cpp
// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("input", &output_tensor));
```
`Rendezvous`负责输入和输出，`Executor`负责从`Rendezvous`取数据执行，会将结果输出到`Rendezvous`(??)，最后`Rendezvous`接收结果到目标tensor。

`Executor`构造器需要设备信息和数据流图。执行时需要一些执行参数，输入数据源(`Rendezvous`)。

当前`Executor`的派生类构造需要`LocalExecutorParams`，包装device和其它一些信息。
```cpp
struct LocalExecutorParams {
  Device* device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library = nullptr;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  std::function<void(OpKernel*)> delete_kernel;

  Executor::Args::NodeOutputsCallback node_outputs_cb;
};
::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      const Graph* graph, Executor** executor);
```

需要注意的是，一个`Executor`执行的其实是一个partition graph，也即是graph的一部分(subgraph)，它的输入可能来自于其它graph执行后的输出。

# tensorflow-0.12 代码阅读

代码目录结构: （主要列出C++层面的代码目录)
tensorflow
 - c  // c API
 - cc  // C++ API
 - core
   - common_runtime
   - distributed_runtime
   - framework
   - graph
   - kernels
   - lib
   - ops
   - platform
   - protobuf
   - public
   - user_ops
   - util
 - stream_executor
   - cuda
   - lib
   - platform

tensorflow执行流程是一个Session作为起点
在单机版本中，它对应的是`DirectSession`. 这里不讨论多机版本（分布式)

类`DirectSession`定义在tensorflow\core\common_runtime\direct_session.h中
它继承于父类`Session`，从而需要实现父类的一些接口.

我们通常并不是直接实例化一个`DirectSession`，而是借助于工厂函数:
`Status NewSession(const SessionOptions& options, Session** out_session)`
```cpp
Status NewSession(const SessionOptions& options, Session** out_session) {
  SessionFactory* factory;
  Status s = SessionFactory::GetFactory(options, &factory);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << s;
    return s;
  }
  *out_session = factory->NewSession(options);
  if (!*out_session) {
    return errors::Internal("Failed to create session.");
  }
  return Status::OK();
}
```
通过SessionOptions来获得一个工厂类，然后用工厂对象`factory`来创建一个session。

```cpp
static mutex* get_session_factory_lock() {
  static mutex session_factory_lock;
  return &session_factory_lock;
}

typedef std::unordered_map<string, SessionFactory*> SessionFactories;
SessionFactories* session_factories() {
  static SessionFactories* factories = new SessionFactories;
  return factories;
}
```
2个全局变量都通过函数获取，线程安全而且没有模块依赖性。

通常客户端代码会这样写：
```cpp
Session* session = nullptr;
Status s = NewSession(option, &session);
....
session->Close();
delete session;
```
所以tensorflow自己并不own这个Session对象的。但是tensorflow内部会维护一个列表，那是所有通过工厂函数创建出来的Session裸指针，这个列表就在Session工厂对象里。
每个Session关闭或销毁时会从这个列表里去除它自己。
```cpp
    std::vector<Device*> devices;
    Status s = DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices);
    if (!s.ok()) {
      LOG(ERROR) << s;
      return nullptr;
    }

    DirectSession* session =
        new DirectSession(options, new DeviceMgr(devices), this);
    {
      mutex_lock l(sessions_lock_);
      sessions_.push_back(session);
    }
    return session;
```
```cpp
::tensorflow::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  {
    mutex_lock l(closed_lock_);
    if (closed_) return ::tensorflow::Status::OK();
    closed_ = true;
  }
  if (factory_ != nullptr) factory_->Deregister(this);  // *****
  return ::tensorflow::Status::OK();
}
```

* 全局线程池
```cpp
thread::ThreadPool* NewThreadPoolFromSessionOptions(
    const SessionOptions& options) {
  const int32 num_threads = NumInterOpThreadsFromSessionOptions(options);
  VLOG(1) << "Direct session inter op parallelism threads: " << num_threads;
  return new thread::ThreadPool(options.env, "Compute", num_threads);
}
...
thread::ThreadPool* GlobalThreadPool(const SessionOptions& options) {
  static thread::ThreadPool* const thread_pool =
      NewThreadPoolFromSessionOptions(options);
  return thread_pool;
}
```
关于session的线程池的配置：DirectSession的构造函数
```cpp
  if (options_.config.session_inter_op_thread_pool_size() > 0) {
    for (int i = 0; i < options_.config.session_inter_op_thread_pool_size();
         ++i) {
      thread_pools_.push_back(NewThreadPoolFromThreadPoolOptions(
          options_, options_.config.session_inter_op_thread_pool(i), i));
    }
    owns_thread_pools_ = true;
  } else if (options_.config.use_per_session_threads()) {
    thread_pools_.push_back(NewThreadPoolFromSessionOptions(options_));
    owns_thread_pools_ = true;
  } else {
    thread_pools_.push_back(GlobalThreadPool(options));
    owns_thread_pools_ = false;
  }
```
一般来说，所有的DirectSession共享一个全局的线程池。

* 设备管理器: `DeviceMgr`


* `ExecutorsAndKeys`
```cpp
  struct ExecutorsAndKeys {
    int64 step_count = 0;
    std::unique_ptr<Graph> graph;
    NameNodeMap name_to_node;
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
    std::vector<PerPartitionExecutorsAndLib> items;
    std::unordered_map<string, string> input_keys;
    std::unordered_map<string, string> output_keys;
  };
```

* `IntraProcessRendezvous`

```cpp
  // Send inputs.
  TF_RETURN_IF_ERROR(SendInputs(inputs, executors_and_keys, run_state.rendez));
  ...

  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  WaitForNotification(&run_state, &step_cancellation_manager,
                      run_options.timeout_in_ms() > 0
                          ? run_options.timeout_in_ms()
                          : operation_timeout_in_ms_);
```

如何创建executors??
```cpp
      GetOrCreateExecutors(pool, input_tensor_names, output_names, target_nodes,
                           &executors_and_keys, &run_state_args);
```
DirectSession维护这样一个cache
```cpp
  mutex executor_lock_;  // protects executors_
  // Holds mappings from signature to the executors that process
  // it. The reason for a level of indirection around mapped_type is
  // to guarantee address stability.
  std::unordered_map<string, std::unique_ptr<ExecutorsAndKeys>> executors_
      GUARDED_BY(executor_lock_);
```

```CPP
  // We create one executor and its dependent library runtime for
  // every partition.
  struct PerPartitionExecutorsAndLib {
    Graph* graph = nullptr;
    std::unique_ptr<FunctionLibraryRuntime> flib;
    std::unique_ptr<Executor> executor;
  };
```
我们还有一个`FunctionLibraryDefinition`
tensorflow\core\framework\function.h
基本上这个类就是个map类，存储着op name => 它的函数定义的映射(container)
```cpp
  const OpRegistryInterface* const default_registry_;
  gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>, HashStr>
      function_defs_;
  gtl::FlatMap<string, string, HashStr> func_grad_;
```

`RunMetadata`  ???


```cpp
    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());
    item->flib.reset(NewFunctionLibraryRuntime(
        device_mgr_.get(), options_.env, device, graph_def_version,
        ek->flib_def.get(), optimizer_opts));
```
```cpp
    item->executor = nullptr;
    Executor* executor;
    TF_RETURN_IF_ERROR(
        NewLocalExecutor(params, iter->second.release(), &executor));
    item->executor.reset(executor);
```
==>>
tensorflow\core\common_runtime\executor.cc

```cpp
Status NewLocalExecutor(const LocalExecutorParams& params, const Graph* graph,
                        Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params, graph);
  Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}
```
```cpp
void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  (new ExecutorState(args, this))->RunAsync(done);
}
```

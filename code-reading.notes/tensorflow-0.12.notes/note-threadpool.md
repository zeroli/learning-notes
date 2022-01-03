# thread::ThreadPool

tensorflow\core\lib\core\threadpool.h

采用pimpl技术来实现这个类.

```cpp
ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads) {
  CHECK_GE(num_threads, 1);
  impl_.reset(
      new ThreadPool::Impl(env, thread_options, "tf_" + name, num_threads));
}
```

Impl实现类是直接继承于一个eigen库提供的线程池类`Eigen::ThreadPoolTempl`:
```cpp
struct ThreadPool::Impl : Eigen::ThreadPoolTempl<EigenEnvironment> {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads)
      : Eigen::ThreadPoolTempl<EigenEnvironment>(
            num_threads, EigenEnvironment(env, thread_options, name)) {}

  void ParallelFor(int64 total, int64 cost_per_unit,
                   std::function<void(int64, int64)> fn) {
    CHECK_GE(total, 0);
    CHECK_EQ(total, (int64)(Eigen::Index)total);
    Eigen::ThreadPoolDevice device(this, this->NumThreads());
    device.parallelFor(
        total, Eigen::TensorOpCost(0, 0, cost_per_unit),
        [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
  }
};
```
eigen类是一个模板类，要求模板参数提供几个接口函数:
* 如何创建线程？？
```cpp
  EnvThread* CreateThread(std::function<void()> f) {
    return env_->StartThread(thread_options_, name_, [=]() {
      // Set the processor flag to flush denormals to zero
      port::ScopedFlushDenormal flush;
      // Set the C++ rounding mode to ROUND TO NEAREST
      port::ScopedSetRound round;
      f();
    });
  }
```
```cpp
struct EigenEnvironment {
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  // 对一个TaskImpl std::unique_ptr的封装
  // 它只有move语义！！
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };
  ...
```

* 如何创建task??
```cpp
  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (port::Tracing::IsActive()) {
      id = port::Tracing::UniqueId();
      port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
                                 id);
    }
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f), Context(ContextKind::kThread), id,
        }),
    };
  }
```
* 如何运行task??
```cpp
  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    if (t.f->trace_id != 0) {
      port::Tracing::ScopedActivity region(
          port::Tracing::EventCategory::kRunClosure, t.f->trace_id);
      t.f->f();
    } else {
      t.f->f();
    }
  }
```

tensorflow\core\common_runtime\process_util.cc
In C++11，如果我们想要move一个对象到一个lambda表达式中，似乎无法直接通过capture clause。但是可以通过如下方式：
```cpp
void SchedClosure(std::function<void()> closure) {
  if (port::Tracing::IsActive()) {
    const uint64 id = port::Tracing::UniqueId();
    port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
                               id);
    std::function<void()> wrapper = std::bind(
        [id](std::function<void()> closure) {
          port::Tracing::ScopedActivity region(
              port::Tracing::EventCategory::kRunClosure, id);
          closure();
        },
        std::move(closure));
    Env::Default()->SchedClosure(std::move(wrapper));
  } else {
    Env::Default()->SchedClosure(std::move(closure));
  }
}
```
> bind一个lambda表达式，然后move那个对象作为lambda参数传入进去。

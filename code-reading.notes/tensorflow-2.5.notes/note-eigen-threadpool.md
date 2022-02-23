# Eigen thread pool

file: eigen\unsupported\Eigen\CXX11\ThreadPool

定义了2中thread pool的实现方式：
* `NonBlockingThreadPool` （默认的，没有EIGEN_USE_SIMPLE_THREAD_POOL编译宏时）
* `SimpleThreadPool`
2个类是子类，实现了接口类`ThreadPoolInterface`的几个接口函数：
```c++
virtual void Schedule(std::function<void()> fn);
virtual void Cancel();
virtual int NumThreads() const;
virtual int CurrentThreadId() const;
virtual ~ThreadPoolInterface()
```

thread pool类模板，要求客户端提供Environment模板参数，这个模板参数类必须定义下面的一些内嵌类和接口函数，以便线程池类进行调用：
1. 提供内嵌类：`Task`，可复制和拷贝；
2. 提供内嵌类：`EnvThread`，线程对象类；
3. 提供创建线程对象的接口函数：
   `EnvThread* CreateThread(std::function<void()> f)`
4. 提供创建task对象的接口函数：
   `Task CreateTask(std::function<void()> f)`
5. 提供执行task对象的接口函数：
   `void ExecuteTask(const Task& t)`

参考版本就是类：`Eigen::StlThreadEnvironment`，基于`std::thread`来实现线程类，其中`Task`直接封装一个lambda表达式：
file: unsupported\Eigen\CXX11\src\ThreadPool\ThreadEnvironment.h
```c++
struct Task {
    std::function<void()> f;
  };
```

`SimpleThreadPool`
===
file：unsupported\Eigen\CXX11\src\ThreadPool\SimpleThreadPool.h
这个简易版本的thread pool基于task的`FIFO`机制来执行task，因此里面会有一个centralized的queue来保存所有的task，线程池里的线程从其中获取task执行。
有一个特别指出就是它提供了`waiters`列表为维护有多少等待的线程，每次一旦有task需要执行，都是显式的让最后一个thread waiter来执行这个新的task。
```c++
struct Waiter {
    std::condition_variable cv;
    Task task;
    bool ready;
  };
```
而不是直接broadcast，让所有的睡眠的线程苏醒，然后其中一个线程获取锁，然后从queue中获取task执行。
这样的设计，应该是考虑到性能问题，再者想让后等待的线程先执行task，实现Last-In-First-Run的效果。

`NonBlockingThreadPool`
===
file：unsupported\Eigen\CXX11\src\ThreadPool\NonBlockingThreadPool.h
上面的简易版本的thread pool只有一个centralized task queue，线程获取queue的task时，需要先获得锁，因此会有锁的竞争。
这个nonBlocking的线程池设计每一线程会有一个task queue (thread localized)，每个线程的线程函数首先从自己的local task queue获取task（不需要锁），如果没有，尝试从其它thread的task queue偷取(Steal) task来执行，如果偷取不到，最后才等待task。
这个类的设计中，都没有看到一个全局的mutex。每个queue都是线程安全的，即使一个线程从另一个线程的queue中偷取task。
偷取时，采用随机的方法从另一个线程的task queue获取task，总共尝试num thread次数。

比较难看懂的是函数`bool WaitForWork(EventCount::Waiter* waiter, Task* t)` (==TODO==)
```c++
  // WaitForWork blocks until new work is available (returns true), or if it is
  // time to exit (returns false). Can optionally return a task to execute in t
  // (in such case t.f != nullptr on return).
  bool WaitForWork(EventCount::Waiter* waiter, Task* t) {
    eigen_assert(!t->f);
    // We already did best-effort emptiness check in Steal, so prepare for
    // blocking.
    ec_.Prewait(waiter);
    // Now do a reliable emptiness check.
    int victim = NonEmptyQueueIndex();
    if (victim != -1) {
      ec_.CancelWait(waiter);
      if (cancelled_) {
        return false;
      } else {
        *t = queues_[victim]->PopBack();
        return true;
      }
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    blocked_++;
    if (done_ && blocked_ == num_threads_) {
      ec_.CancelWait(waiter);
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (NonEmptyQueueIndex() != -1) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        blocked_--;
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(true);
      return false;
    }
    ec_.CommitWait(waiter);
    blocked_--;
    return true;
  }
```
上面的`ec_`是类`EventCount`的对象。

```c++
// Steal tries to steal work from other worker threads in best-effort manner.
  Task Steal() {
    PerThread* pt = GetPerThread();
    const size_t size = queues_.size();
    unsigned r = Rand(&pt->rand);
    unsigned inc = coprimes_[r % coprimes_.size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      Task t = queues_[victim]->PopBack();
      if (t.f) {
        return t;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return Task();
  }
```

一个小小的优化路径就是当线程池只有一个线程时，并不需要进行偷取，当前queue没有task的话，就直接等待。

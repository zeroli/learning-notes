# ExecutorBarrier

```cpp
// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  ExecutorBarrier(int num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}

  ~ExecutorBarrier() {}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  StatusCallback Get() {
    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  int pending_ GUARDED_BY(mu_) = 0;
  Status status_ GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
    bool error = false;
    Rendezvous* error_rendez = nullptr;
    StatusCallback done = nullptr;
    Status status;
    {
      mutex_lock l(mu_);
      // If we are the first error encountered, mark the status
      // appropriately and later trigger an abort of the Rendezvous
      // object by this thread only.
      if (status_.ok() && !s.ok()) {
        error = true;
        error_rendez = rendez_;
        error_rendez->Ref();
        status_ = s;
      }

      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        done = done_cb_;
        done_cb_ = nullptr;
      }

      status = status_;
    }

    if (error) {
      error_rendez->StartAbort(status);
      error_rendez->Unref();
    }
    if (done != nullptr) {
      delete this;
      done(status);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};
```

每个线程完成之后，不管是正常done，还是aborted，都会调用到WhenDone callback，这时候Barrier就会进行处理，记录状态，直到最后一个线程调用callback。
一旦发现有线程运行出现错误，便会启动abort代码，最后一个线程完成，便会销毁Barrier自己，然后调用客户端在初始化Barrier时传入进来的callback，通知客户端最终的状态。
这段代码为了尽量减少mutex竞争的代码范围，运用了swap技术，将成员变量转移到局部变量，尽量避免锁太长时间的问题。而且done callback也是在锁之外调用的。

```cpp
  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }
```
```cpp
void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  (new ExecutorState(args, this))->RunAsync(done);
}
```
Executor的同步版本，对异步版本进行包装，传入一个callback通知外围的等待者。
```cpp
  // Synchronous wrapper for RunAsync().
  Status Run(const Args& args) {
    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};
```

看这里，ExecutorImpl需要很多参数

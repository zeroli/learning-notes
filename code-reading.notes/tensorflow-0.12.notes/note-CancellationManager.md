# CancellationManager

涉及到的文件如下：
```SH
tensorflow\core\framework\cancellation.h
tensorflow\core\framework\cancellation.cc
```

上面文件只定义了`CancellationManager`类，提供的接口如下：
```CPP
void StartCancel();
bool IsCancelled();
CancellationToken get_cancellation_token();
bool RegisterCallback(CancellationToken token, CancelCallback callback);
bool DeregisterCallback(CancellationToken token);
```

这个类的作用主要是提供一个cancellation的管理器，有一个地方可以管理所有需要cancel操作的集合。
比如说我有一个类，在最后需要某人帮忙cancel掉，然后我会自己检测外界是否需要我cancel，然后cancel退出。这样我就可以把我的cancel接口给这个类进行管理。
一旦在程序某个时候需要cancel了，就会调用`CancellationManager`的`StartCancel`，这个函数就会调用所有注册的callbacks，之后进行通知。
```CPP
void CancellationManager::StartCancel() {
  gtl::FlatMap<CancellationToken, CancelCallback> callbacks_to_run;
  {
    mutex_lock l(mu_);
    if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
      return;
    }
    is_cancelling_ = true;
    std::swap(callbacks_, callbacks_to_run);
  }
  // We call these callbacks without holding mu_, so that concurrent
  // calls to DeregisterCallback, which can happen asynchronously, do
  // not block. The callbacks remain valid because any concurrent call
  // to DeregisterCallback will block until the
  // cancelled_notification_ is notified.
  for (auto key_and_value : callbacks_to_run) {
    key_and_value.second();
  }
  {
    mutex_lock l(mu_);
    is_cancelling_ = false;
    is_cancelled_.store(true, std::memory_order_release);
  }
  cancelled_notification_.Notify();
}
```
注册callback时需要用它的接口函数获得一个`CancellationToken`。
```CPP
bool CancellationManager::RegisterCallback(CancellationToken token,
                                           CancelCallback callback) {
  mutex_lock l(mu_);
  CHECK_LT(token, next_cancellation_token_) << "Invalid cancellation token";
  bool should_register = !is_cancelled_ && !is_cancelling_;
  if (should_register) {
    std::swap(callbacks_[token], callback);
  }
  return should_register;
}
```
解注册时需要特别注意，因为`CancellationManager`有可能正在进行Cancel，正在调用所有的callbacks
```CPP
bool CancellationManager::DeregisterCallback(CancellationToken token) {
  mu_.lock();
  if (is_cancelled_) {
    mu_.unlock();
    return false;
  } else if (is_cancelling_) {
    mu_.unlock();
    // Wait for all of the cancellation callbacks to be called. This
    // wait ensures that the caller of DeregisterCallback does not
    // return immediately and free objects that may be used in the
    // execution of any currently pending callbacks in StartCancel.
    cancelled_notification_.WaitForNotification();
    return false;
  } else {
    callbacks_.erase(token);
    mu_.unlock();
    return true;
  }
}
```
所以我们需要等待cancel操作完全结束，因此这个解注册函数有可能会block。


在这里顺便把`Notification`也介绍下。
这个类简单包装了线程间等待和通知机制，因而它提供的接口如下：
```CPP
  void Notify() {
    mutex_lock l(mu_);
    assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }
  void WaitForNotification() {
    mutex_lock l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }
```
故而实现需要有Mutex和ConditionVariable的组合。
但是从它的设计来看，这也是一种抽象，用抽象来表达想法，对一些事物进行封装，从而达到代码重用的目的。
这个类完全可以被用在任何需要这种消息通知机制的代码逻辑里面，而不用自己每次需要去重新实现。
但是如果一段代码的Mutex需要保护的数据更多，那就需要另外的Mutex来进行保护了，故而Mutex就用的更多。

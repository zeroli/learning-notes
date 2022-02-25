# protobuf源码阅读

google init once的实现
=====

google自己实现了一套类似std::once的init once，调用`GoogleOnceInit`时传入一个全局的`ProtobufOnceType`类型，和一个初始化执行函数。`ProtobufOnceType`具备原子性。

```c++
inline void GoogleOnceInit(ProtobufOnceType* once, void (*init_func)()) {
  if (internal::Acquire_Load(once) != ONCE_STATE_DONE) {
    internal::FunctionClosure0 func(init_func, false);
    GoogleOnceInitImpl(once, &func);
  }
}
```

```c++
void GoogleOnceInitImpl(ProtobufOnceType* once, Closure* closure) {
  internal::AtomicWord state = internal::Acquire_Load(once);
  // Fast path. The provided closure was already executed.
  if (state == ONCE_STATE_DONE) {
    return;
  }
  // The closure execution did not complete yet. The once object can be in one
  // of the two following states:
  //   - UNINITIALIZED: We are the first thread calling this function.
  //   - EXECUTING_CLOSURE: Another thread is already executing the closure.
  //
  // First, try to change the state from UNINITIALIZED to EXECUTING_CLOSURE
  // atomically.
  state = internal::Acquire_CompareAndSwap(
      once, ONCE_STATE_UNINITIALIZED, ONCE_STATE_EXECUTING_CLOSURE);
  if (state == ONCE_STATE_UNINITIALIZED) {
    // We are the first thread to call this function, so we have to call the
    // closure.
    closure->Run();
    internal::Release_Store(once, ONCE_STATE_DONE);
  } else {
    // Another thread has already started executing the closure. We need to
    // wait until it completes the initialization.
    while (state == ONCE_STATE_EXECUTING_CLOSURE) {
      // Note that futex() could be used here on Linux as an improvement.
      SchedYield();
      state = internal::Acquire_Load(once);
    }
  }
}
```
`Acquire_CompareAndSwap`原子的比较并修改第一个参数，如果第一个参数并不是我们期望的值，返回第一个参数的旧值。
* 如果一个线程在执行时，是第一个线程，那么可以安全的执行用户的函数，然后修改once值为done；
* 如果其它线程也执行到这里了，那么不断重试`SchedYield`，判断`once`的最新值是否为done。
* 同时上述方式避免mutex锁的抢占（其实也可以用mutex锁来达到同样的目的）。

```c++
void SchedYield() {
#ifdef _WIN32
  Sleep(0);
#else  // POSIX
  sched_yield();
#endif
}
```

https://man7.org/linux/man-pages/man2/sched_yield.2.html
> sched_yield() causes the calling thread to relinquish the CPU.
       The thread is moved to the end of the queue for its static
       priority and a new thread gets to run.
In the Linux implementation, sched_yield() always succeeds

`sched_yield`和`sleep(0)`的区别（@linux)：
* They are essentially same. Both of them take the thread off the core. Only difference is process calling sched_yield() might get scheduled on other core immediately whereas the thread calling sleep with higher sleep value (>0) will sleep at least that much duration. sleep(0) and sched_yield() are one and the same. (==????==)

* From the FreeBSD manual pages:

The sched_yield() system call forces the running process to relinquish the processor until it again becomes the head of its process list.
The sleep() function suspends execution of the calling process until either seconds seconds have elapsed or a signal is delivered to the process and its action is to invoke a signal-catching function or to terminate the process. System activity may lengthen the sleep by an indeterminate amount.
I had a quick look to see what zero would do and it does appear the the system call is executed, but not sure if that is sufficient to allow for a task/thread switch.

* sleep sends the current thread to the "wait" state but sched_yeild() will take it to the "ready" state & relinquish the CPU

建议用`sched_yield`来实现让出CPU，而不是sleep(0)，或者usleep(0)；在linux上。
windows上只能用`Sleep(0)`，而且跟linux上的`sched_yield`效果一样。
参见文章：
`sleep(0)、usleep(0)与sched_yield() 调度 - schips - 博客园.pdf`

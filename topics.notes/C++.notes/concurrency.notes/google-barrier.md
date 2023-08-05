# 下面这个是google 在eigen库中的barrier的简易实现（unsupported\Eigen\CXX11\src\Tensor\TensorDeviceThreadPool.h）
* 采用bit-0来区分是否有waiter waiting比较巧妙，从而避免了可能的mutex lock/unlock/notify_all操作
* 在wait中，如果所有的count down都到了0，也不需要mutex lock/wait了

```c++
struct Barrier {
    explicit Barrier(unsigned int count)
        : d_state(count << 1), d_notified(false)
    {
        // check overflow?
        assert(((count << 1) >> 1) == count);
    }
    ~Barrier（）
    {
        // check if all threads count down to 0
        assert((d_state >> 1) == 0);
    }
    void CountDown(unsigned int x = 1)
    {
        unsigned int newval = d_state.fetch_sub((x << 1), std::memory_order_acq_rel) - (x << 1);
        if (newval != 1) {
            // assert: (newval + (x << 1)) & ~1 == 0
            // no waiter waiting, or count has not dropped to 0 yet (+waiter waiting already)
            // xxxx0, 0000, or xxxx1
            return;
        }
        // newval == 1, which means there is waiter waiting (at least one) and all count down to 0
        {
            std::lock_guard<std::mutex> lock(d_mtx);
            d_notified = true;
        }
        d_cv.notify_all();
    }

    void Wait()
    {
        unsigned int val = d_state.fetch_or(1, std::memory_order_acq_rel);
        if ((val >> 1) == 0) {  // already count down to 0
            return;
        }
        std::unique_lock<std::mutex> lock(d_mtx);
        while (!d_notified) {
            d_cv.wait(lock);
        }
    }
private:
    std::mutex d_mtx;
    std::condition_variable d_cv;
    // lower-bit flags if there is waiter waiting
    // other bits flags the real count
    std::atomic<unsigned int> d_state;
    bool d_notified;
};

```
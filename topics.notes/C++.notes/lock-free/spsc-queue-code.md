# single-producer, single-consumer, queue两种不同实现
```c++
template <typename T>
struct SPSCQueue1 {
    explicit SPSCQueue1(int capacity)
        : d_data(static_cast<T*>(malloc(sizeof(T) * (capacity + 1))))
        , d_capacity(capacity + 1)
    {
           
    }
    ~SPSCQueue1()
    {
        free(d_data);
    }
    SPSCQueue1(const SPSCQueue1&) = delete;
    SPSCQueue1& operator=(const SPSCQueue1&) = delete;

    // return true if push successfuly, otherwise return false
    bool push(T t)
    {
        // d_write + 1 == d_read => full
        // ?????????????,????????,??relaxed
        int cur_write = d_write.load(std::memory_order_relaxed);  
        int next_write = next(cur_write);
        // ?????????d_read,???????????,????acquired
        if (next_write == d_read.load(std::memory_order_acquired)) {
            return false;
        }
        // 3?????4????,??out-of-order,??4??relase
        // ?????????????????????,
        // ???????????acquire???????????????????????
        // ??4?????????????,??3
        new (d_data + cur_write) T(std::move(t));  // 3
        d_write.store(next_write, std::memory_order_release);  // 4
        return true;
    }
    // return true if pop successfully, and returned in `t`
    // otherwise, return false, `t` not changed
    bool pop(T& t)
    {
        // d_write = d_read => empty
        // ?????????????,????????,??relaxed
        int cur_read = d_read.load(std::memory_order_relaxed);
        // ?????????d_write,???????????,????acquired
        // ???????????????????,?????????????
        if (cur_read == d_write.load(std::memory_order_acquired)) {
            return false;
        }
        // ??????????,d_read??,???????d_read??????
        t = std::move(d_data[cur_read]);  // 3 
        // ????release????
        d_read.store(next(cur_read), std::memory_order_release); // 4
        return true;
    }
    
    bool wasEmpty() const
    {
        return d_write.load() == d_read.load();
    }
    bool wasFull() const
    {
        return next(d_write.load()) + 1 == d_read.load();
    }
private:
    int next(int pos)
    {
        return (pos + 1) % (d_capacity + 1);
    }
private:
    // cache line 64bytes??,??false-sharing
    alignas(64) std::atomic<int> d_read;
    alignas(64) std::atomic<int> d_write;
    
    T* d_data;
    int d_capacity;
};


// ????????,??d_size?????????
template <typename T>
struct SPSCQueue2 {
    explicit SPSCQueue2(int capacity)
        : d_read(0), d_write(0)
        , d_data(static_cast<T*>(malloc(sizeof(T) * (capacity))))
        , d_capacity(capacity), d_size(0)
    {
           
    }
    ~SPSCQueue1()
    {
        free(d_data);
    }
    SPSCQueue1(const SPSCQueue1&) = delete;
    SPSCQueue1& operator=(const SPSCQueue1&) = delete;

    // return true if push successfuly, otherwise return false
    bool push(T t)
    {
        if (wasFull()) {
            return false;
        }
        
        new (d_data + d_write) T(std::move(t));
        d_write = next(d_write);
        d_size.fetch_add(1, std::memory_order_release);
        return true;
    }
    // return true if pop successfully, and returned in `t`
    // otherwise, return false, `t` not changed
    bool pop(T& t)
    {
        if (wasEmpty()) {
            return false;
        }
        
        t = std::move(d_data[d_read]);
        d_read = next(d_read);
        d_size.fetch_sub(1, std::memory_order_release);
        return true;
    }
    
    bool wasEmpty() const
    {
        return d_size.load() == 0;
    }
    bool wasFull() const
    {
        return d_size.load() == d_capacity;
    }
private:
    int next(int pos)
    {
        return (pos + 1) % d_capacity;
    }
private:
    alignas(64) int d_read;
    alignas(64) int d_write;
    T* d_data;
    const int d_capacity;
    
    std::atomic<int> d_size;
};
```

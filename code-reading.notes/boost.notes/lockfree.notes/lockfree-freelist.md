# boost lockfree freelist的实现

这个类并没有public，位于detail/目录下，提供给lockfree stack/queue作为底层的pool采用。它的底层仍然采用lockfree的方式实现。

* 类`freelist_stack`
基于链表实现的stack (FILO机制）的freelist (这种方式比较常见)。

```c++
    struct freelist_node
    {
        tagged_ptr<freelist_node> next;
    };

    typedef tagged_ptr<freelist_node> tagged_node_ptr;
```
* 基于单向链表；
* 当客户数据回收时，直接在客户数据内存中定义`next`，从而利用用户数据空间；
* `tagged_ptr`是对一个指针的简单封装，支持指针压缩。


# 根据请求的freelist的大小`n`，预分配`n`个node，allocate/deallocate。
```c++
    template <typename Allocator>
    freelist_stack (Allocator const & alloc, std::size_t n = 0):
        Alloc(alloc),
        pool_(tagged_node_ptr(NULL))
    {
        for (std::size_t i = 0; i != n; ++i) {
            T * node = Alloc::allocate(1);
#ifdef BOOST_LOCKFREE_FREELIST_INIT_RUNS_DTOR
            destruct<false>(node);
#else
            deallocate<false>(node);
#endif
        }
    }
```

# 从freelist中取出一个node，然后运用placement new进行构造
```c++
    template <bool ThreadSafe, bool Bounded>
    T * construct (void)
    {
        T * node = allocate<ThreadSafe, Bounded>();
        if (node)
            new(node) T();
        return node;
    }
```

# allocate节点，线程不安全的方式
```c++
    template <bool Bounded>
    T * allocate_impl_unsafe (void)
    {
        tagged_node_ptr old_pool = pool_.load(memory_order_relaxed);

        if (!old_pool.get_ptr()) {
            if (!Bounded)
                return Alloc::allocate(1);
            else
                return 0;
        }

        freelist_node * new_pool_ptr = old_pool->next.get_ptr();
        tagged_node_ptr new_pool (new_pool_ptr, old_pool.get_next_tag());

        pool_.store(new_pool, memory_order_relaxed);
        void * ptr = old_pool.get_ptr();
        return reinterpret_cast<T*>(ptr);
    }
```
* 全部采用`memory_order_relaxed`的方式进行内存同步

# 接下来看下，线程(lockfree)安全的方式从freelist头节点取出一个node:
```c++
    template <bool Bounded>
    T * allocate_impl (void)
    {
        tagged_node_ptr old_pool = pool_.load(memory_order_consume);  // 采用`memory_order_consume`

        for(;;) {
            if (!old_pool.get_ptr()) {
                if (!Bounded)
                    return Alloc::allocate(1);
                else
                    return 0;
            }

            freelist_node * new_pool_ptr = old_pool->next.get_ptr();
            tagged_node_ptr new_pool (new_pool_ptr, old_pool.get_next_tag());
            // `weak`的方式: CAS
            if (pool_.compare_exchange_weak(old_pool, new_pool)) {
                void * ptr = old_pool.get_ptr();
                return reinterpret_cast<T*>(ptr);
            }
        }
    }
```
* 这样的操作方式比较典型


# deallocate线程不安全的方式放回一个node
```c++
    void deallocate_impl_unsafe (T * n)
    {
        void * node = n;
        tagged_node_ptr old_pool = pool_.load(memory_order_relaxed);
        freelist_node * new_pool_ptr = reinterpret_cast<freelist_node*>(node);

        tagged_node_ptr new_pool (new_pool_ptr, old_pool.get_tag());
        new_pool->next.set_ptr(old_pool.get_ptr());

        pool_.store(new_pool, memory_order_relaxed);
    }
```

# 接下来是线程安全的方式(lockfree):
```c++
    void deallocate_impl (T * n)
    {
        void * node = n;
        tagged_node_ptr old_pool = pool_.load(memory_order_consume);  // `memory_order_consume`
        freelist_node * new_pool_ptr = reinterpret_cast<freelist_node*>(node);

        for(;;) {
            tagged_node_ptr new_pool (new_pool_ptr, old_pool.get_tag());
            new_pool->next.set_ptr(old_pool.get_ptr());

            if (pool_.compare_exchange_weak(old_pool, new_pool))
                return;
        }
    }
```

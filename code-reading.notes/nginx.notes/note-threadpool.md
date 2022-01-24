# NGINX的线程池实现摘要

```c++
struct thread_task_t {
    task_t* next;
    void* ctx;
    void (*handler)(void* ctx);
};

struct thread_pool_queue {
    thread_task_t* first;
    thread_task_t** last;
};

#define thread_pool_queue_init(q) \
    q->first = nullptr; \
    q->last = &(q->first); \

// add one task to tail of the queue
*tp->queue->last = task;
tp->queue->last = &(task->next);
```
上述代码的queue是一个单链表，但是实现却非常巧妙，其中`last`是一个2级指针，初始化时指向`q->first`，之后每增加一个task，直接指向`task->next`。如此处理，我们就不需要判断first或者last指针是否为空的情况了。

# FIFO queue的学习

FIFO queue
=====
- 这个实现不保证线程安全；
- 这个类实现了基于FIFO (first-in-first-out)策略的循环队列
  - head和tail用来track当前队列头和尾;
  - head==tail时，指示队列为空；
  - head==tail+1时，指示队列为满，它会自动grow size，初始值为32
- 提供2个接口函数: push/pop
- 数据存储类型为void*，因此pop时，如果队列为空，返回一个(void*)0值，或者对应的空值；
- 这个类跟task_run_handle绑定起来的

```CPP {.numberLines}
	~fifo_queue()
	{
		// Free any unexecuted tasks
		for (std::size_t i = head; i != tail; i = (i + 1) & (items.size() - 1))
			task_run_handle::from_void_ptr(items[i]);
	}
// Push a task to the end of the queue
	void push(task_run_handle t)
	{
		// Resize queue if it is full
		if (head == ((tail + 1) & (items.size() - 1))) {
			detail::aligned_array<void*, LIBASYNC_CACHELINE_SIZE> new_items(items.size() * 2);
			for (std::size_t i = 0; i != items.size(); i++)
				new_items[i] = items[(i + head) & (items.size() - 1)];
			head = 0;
			tail = items.size() - 1;
			items = std::move(new_items);
		}

		// Push the item
		items[tail] = t.to_void_ptr();
		tail = (tail + 1) & (items.size() - 1);
	}

	// Pop a task from the front of the queue
	task_run_handle pop()
	{
		// See if an item is available
		if (head == tail)
			return task_run_handle();
		else {
			void* x = items[head];
			head = (head + 1) & (items.size() - 1);
			return task_run_handle::from_void_ptr(x);
		}
	}
```
- push时如果队列满，则doubling的增加queue的大小；
- 析构时，会cancel所有未执行的task：`task_run_handle::from_void_ptr(items[i]);`;
    ```CPP {.numberLines}
    static task_run_handle from_void_ptr(void* ptr)
	{
		return task_run_handle(detail::task_ptr(static_cast<detail::task_base*>(ptr)));
	}
    ```
    > 这个函数构造一个临时的task_run_handle然后返回丢掉。

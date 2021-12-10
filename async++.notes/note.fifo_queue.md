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

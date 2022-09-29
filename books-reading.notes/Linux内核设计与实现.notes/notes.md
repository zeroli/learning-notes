# Linux内核设计与实现(第3版)

- 重读第12章内存管理
    - zone/page，Linux是如何管理物理内存
    - kmalloc/vmalloc，分配物理内存页面和虚拟内存页面
    - slab allocator，小对象物理内存分配
- 阅读第15章进程地址空间
    - mm_struct结构体描述进程内存布局
        - vm_area_struct结构体链表和红黑树共同描述所有虚拟内存区域
        - 不同进程可以共享`mm_struct`结构体，users，代表共享内存空间，比如线程
        - mm_struct结构体是从slab内存分配器上分配的，`mm_cachep`对象缓存
        - 内核线程没有mm_struct结构，它的进程struct中对应mm指针为null, 但是active_mm指针指向前一个进程的mm结构，但是因为内核线程并不会访问用户地址空间，不过需要访问内核地址空间

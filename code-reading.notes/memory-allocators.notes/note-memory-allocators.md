# memory-allocators开源库的学习

https://github.com/mtrebi/memory-allocators.git

这个project实现了几种memory allocators:
- `LinearAllocator`: 预先分配一个大的memory block，allocation时通过`offset`来offset+size来分配内存，不需要free，期待着最后一次性的reset释放这个大的memory block;
- `CAllocator`: 简单的基于malloc/free来进行分配和释放；
- `FreeListAllocator`: 典型的基于free list策略进行分配和回收的内存分配器：
  - 同样预先分配一个大的memory block，将这个block赋给free list；
  - freelist基于单链表实现；
  - allocate时从freeslit->head进行cut chunk分配，剩下的链入freelist；
    * 可以基于first-fit查找策略从free list中寻找第一个满足大小需求的chunk;
    * 可以基于best-fit查找策略从free list中寻找最合适大小的chunk;
  - free时，根据释放的指针寻找在free list中的插入位置；
    * freelist的所有节点以chunk地址从小到大的顺序排列；
    * freelist的节点有序就可以前后节点合并；
- `PoolAllocator`： 典型的基于pool/chunksize策略进行分配和回收的内存分配器：
  - 同样预先分配一个大memory block，以chunksize将它分成N个chunks，链入一个stack中；
  - allocate时从stack中pop一个chunk返回；
    * 如果没有空间了，则报错；
  - free时，将直接根据ptr转换为chunk，push到stack中；
  - allocate/free都期待中申请的内存大小是chunk size;
- 上述所有子类都继承于一个基类`Allocator`；

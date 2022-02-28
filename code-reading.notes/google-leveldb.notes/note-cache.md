# LevelDB源码阅读

Cache
===

类`Cache`，提供了一个比较有意思的嵌套类:`Handle`，是空的，作为一个`void*`来使用，但是具有更强的表达力。
```c++
  // Opaque handle to an entry stored in the cache.
  struct Handle {};
  virtual Handle* Insert(const Slice& key, void* value, size_t charge,
                         void (*deleter)(const Slice& key, void* value)) = 0;
```

`Cache`类定义了一些接口，子类`LRUCache`实现这些接口。
key是字符串Slice，value采用void*来表示，类型无关。
接口如下：
* Insert
* Lookup
* Release
* Value
* Erase
* Prune
插入，查找，释放，获取值，擦除

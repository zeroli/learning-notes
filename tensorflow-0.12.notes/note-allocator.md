# 研究TF的allocator的一切事情

* Allocator

```sh
tensorflow\core\framework\allocator.h
tensorflow\core\framework\allocator.cc
```
这是一个抽象基类，它的虚接口函数如下：
- virtual string Name() = 0;
- virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;
- virtual void* AllocateRaw(size_t alignment, size_t num_bytes, const AllocationAttributes& allocation_attr);
- virtual void DeallocateRaw(void* ptr) = 0;

它还提供C++模板函数方便allocate具体类型的C++对象：
```CPP
  // Convenience functions to do typed allocation.  C++ constructors
  // and destructors are invoked for complex types if necessary,
  // depending on the concrete Allocator implementation. May return
  // NULL if the tensor has too many elements to represent in a single
  // allocation.
  template <typename T>
  T* Allocate(size_t num_elements) {
    return Allocate<T>(num_elements, AllocationAttributes());
  }

  template <typename T>
  T* Allocate(size_t num_elements,
              const AllocationAttributes& allocation_attr) {
    // TODO(jeff): Do we need to allow clients to pass in alignment
    // requirements?

    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
      return NULL;
    }
    // default: kAllocatorAlignment = 32
    void* p = AllocateRaw(kAllocatorAlignment, sizeof(T) * num_elements,
                          allocation_attr);
    T* typed_p = reinterpret_cast<T*>(p);
    // 默认自动调用类的构造函数
    if (typed_p) RunCtor<T>(typed_p, num_elements);
    return typed_p;
  }

  template <typename T>
  void Deallocate(T* ptr, size_t num_elements) {
    if (ptr) {
      // 默认自动调用类的析构函数
      RunDtor<T>(ptr, num_elements);
      DeallocateRaw(ptr);
    }
  }
```
对一些简单的基本类型构造函数和析构函数，可以啥都不做：
```CPP
// No constructors or destructors are run for simple types
  template <typename T>
  void RunCtor(T* p, size_t n) {
    static_assert(is_simple_type<T>::value, "T is not a simple type.");
  }

  template <typename T>
  void RunDtor(T* p, size_t n) {}
```
上面的简单类型它并没有用std::is_trivial来判断：
`tensorflow\core\framework\type_traits.h`
```CPP
// is_simple_type<T>::value if T[] can be safely constructed and destructed
// without running T() and ~T().  We do not use std::is_trivial<T>
// directly because std::complex<float> and std::complex<double> are
// not trivial, but their arrays can be constructed and destructed
// without running their default ctors and dtors.
template <typename T>
struct is_simple_type {
  static constexpr bool value =
      std::is_trivial<T>::value || std::is_same<T, Eigen::half>::value ||
      std::is_same<T, complex64>::value || std::is_same<T, complex128>::value ||
      is_quantized<T>::value || std::is_same<T, bfloat16>::value;
};
```
用户自定义的类型可以参照下面的方式来构造：
```CPP
  // Runs string's default constructor for  p[0], p[1], ..., p[n-1].
  virtual void RunStringCtor(string* p, size_t n) {
    for (size_t i = 0; i < n; ++p, ++i) new (p) string();
  }

  // Runs string's default destructor for  p[0], p[1], ..., p[n-1].
  virtual void RunStringDtor(string* p, size_t n) {
    for (size_t i = 0; i < n; ++p, ++i) p->~string();
  }

// 它们的模板具体实现也定义在类的外面：
template <>
inline void Allocator::RunCtor(string* p, size_t n) {
  RunStringCtor(p, n);
}

template <>
inline void Allocator::RunDtor(string* p, size_t n) {
  RunStringDtor(p, n);
}
```

`SubAllocator`的定义：
```CPP
// Abstract interface of an object that does the underlying suballoc/free of
// memory for a higher-level allocator.
class SubAllocator {
 public:
  virtual ~SubAllocator() {}
  virtual void* Alloc(size_t alignment, size_t num_bytes) = 0;
  virtual void Free(void* ptr, size_t num_bytes) = 0;
};
```

`allocator.cc`文件里定义了一个`CPUAllocator`，实现了CPU版本的allocator
```CPP
void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    void* p = port::aligned_malloc(num_bytes, alignment);
    return p;
}
void DeallocateRaw(void* ptr) override {
    port::aligned_free(ptr);
}
```
然后它提供了一个utility函数，用来创建CpuAllocator对象：
```CPP
Allocator* MakeCpuAllocator() {
  Allocator* allocator = new CPUAllocator;
  if (cpu_allocator_collect_full_stats || LogMemory::IsEnabled()) {
    allocator = new TrackingAllocator(allocator, true);
  }
  return allocator;
}
```
 它根据当前的设置可以wrap成不同的allocator，这里是`TrackingAllocator`
这个`TrackingAllocator`有这样一个描述：
>// TrackingAllocator is a wrapper for an Allocator. It keeps a running
// count of the number of bytes allocated through the wrapper. It is
// used by the Executor to "charge" allocations to particular Op
// executions. Each Op gets a separate TrackingAllocator wrapper
// around the underlying allocator.
//
// The implementation assumes the invariant that all calls to
// AllocateRaw by an Op (or work items spawned by the Op) will occur
// before the Op's Compute method returns. Thus the high watermark is
// established once Compute returns.
//
// DeallocateRaw can be called long after the Op has finished,
// e.g. when an output tensor is deallocated, and the wrapper cannot
// be deleted until the last of these calls has occurred.  The
// TrackingAllocator keeps track of outstanding calls using a
// reference count, and deletes itself once the last call has been
// received and the high watermark has been retrieved.

其实我们可以利用这样的设计来实现在运行时对某一个模块或一段代码段进行内存分配的tracking，而且它可以是单线程的。比如下面的伪代码段：
```CPP
{
    MyTrackingAllocator alloc;
    // 需要track的代码段和模块
    ...
}
```
`MyTrackingAllocator`的构造函数可以这样
```CPP
MyTrackingAllocator::MyTrackingAllocator()
    : cur_(alloc::GetcurrentAllocatr())
{
    alloc::SetCurrentAllocator(this);
}
MyTrackingAllocator::~MyTrackingAllocator()
{
    alloc::SetCurrentAllocator(cur_);
}
```
所有的alloc/dealloc全部转调用到`cur_`allocator，在做完一些tracking和统计工作之后。最后在析构函数里恢复原先的allocator
当然我们可以将save/restore做成另外一个RAII的类，外面类实例化这个类。这样TrackingAllocator类只负责进行tracking和统计工作，之后RAII类获取统计信息之后进行统计输出。
这个实现，需要注意多线程的情况，多个线程同时track。每个线程获取的CurrentAllocator都应该是相同的。
如果这样的话，一些全局函数alloc/free用的allocator首先应该从TLS中去拿，然后再从global中获取allocator，如果TLS中没有的话。这个机制类似于我们实现的TID()方法。


* GPUBFCAllocator:
```SH
tensorflow\core\common_runtime\gpu\gpu_bfc_allocator.h
tensorflow\core\common_runtime\gpu\gpu_bfc_allocator.cc
```

这个类是直接继承于`BFCAllocator`类。在实例化这个类，传入`BFCAllocator` 构造函数中的`SubAllocator`是`GpuMemAllocator`类的对象。
所以每个device对应一个`BFCAllocator`类的对象。
`GpuMemAllocator`是继承于`SubAllocatr`类，实现的2个接口如下：
```CPP
  void* Alloc(size_t alignment, size_t num_bytes) override {
    void* ptr = nullptr;
    if (num_bytes > 0) {
      ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      gpu::DeviceMemoryBase gpu_ptr(ptr);
      stream_exec_->Deallocate(&gpu_ptr);
    }
  }
```
转调用到底层的`stream_exec_`进行处理，`stream_exec_`是保存在`GpuMemAllocator`类里面的一个指针，不own。
`GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie()`

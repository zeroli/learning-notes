# process state的学习

```SH
tensorflow\core\common_runtime\gpu\process_state.cc
tensorflow\core\common_runtime\gpu\process_state.h
```

`ProcessState`类是一个singleton类，它抽象出当前进程的状态，里面包含有cpu/gpu的allocator，各种allocator都是从它这里创建的，而且最后销毁，故而TF里面的allocator是全局性的。
```CPP
ProcessState* ProcessState::instance_ = nullptr;

/*static*/ ProcessState* ProcessState::singleton() {
  if (instance_ == nullptr) {
    instance_ = new ProcessState;
  }

  return instance_;
}
```

首先看下CPU allocator:
```CPP
Allocator* ProcessState::GetCPUAllocator(int numa_node) {
  // Although we're temporarily ignoring numa_node, check for legality.
  CHECK_GE(numa_node, 0);
  // TODO(tucker): actually maintain separate CPUAllocators for
  // different numa_nodes.  For now, just one.
  numa_node = 0;
  mutex_lock lock(mu_);
  while (cpu_allocators_.size() <= static_cast<size_t>(numa_node)) {
    Allocator* allocator =
        new PoolAllocator(100 /*pool_size_limit*/, true /*auto_resize*/,
                          new BasicCPUAllocator(), new NoopRounder, "cpu_pool");
    if (LogMemory::IsEnabled()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    cpu_allocators_.push_back(allocator);
  }
  return cpu_allocators_[0];
}
```
`BasicCPUAllocator`类比较简单，直接继承于`SubAllocator`基类，实现`Alloc`和`Free`接口函数。用`aligned_malloc`和`aligned_free`来实现`Alloc`和`Free`，实现内存地址对齐的分配和释放。
`PoolAllocator`是在其基础上实现的pool policy的内存分配器，实现`Allocator`基类的接口函数，供外界调用。`PoolAllocator`是一个与`BFCAllocator`同等地位的allocator。

GPU allocator跟CPU allocator比较类似，同样创建一个per gpu device的BFC Allocator：
```CPP
Allocator* ProcessState::GetGPUAllocator(const GPUOptions& options, int gpu_id,
                                         size_t total_bytes) {
                                             ...
    gpu_allocator = new GPUBFCAllocator(gpu_id, total_bytes, options);

    // If true, checks for memory overwrites by writing
    // distinctive patterns on both ends of allocated memory.
    static const bool kGPUDebug = false;
    if (kGPUDebug) {
      gpu_allocator = new GPUDebugAllocator(gpu_allocator, gpu_id);
      gpu_allocator = new GPUNanResetAllocator(gpu_allocator, gpu_id);
    }
    gpu_allocators_[gpu_id] = gpu_allocator;
    ...
```

最后的问题是，什么时候调用`ProcessState::singleton()`来触发实例化这个全局的`ProcessState`的呢？
譬如`GPUDevice`想要实现接口函数`Allocator* GetAllocator(AllocatorAttributes attr) override`函数，就会调用`ProcessState::singleton`拿到当前进程的ProcessState，从而获取对应设备的allocator。
```CPP
class GPUDevice : public BaseGPUDevice {
    ...
      Allocator* GetAllocator(AllocatorAttributes attr) override {
    if (attr.on_host()) {
      ProcessState* ps = ProcessState::singleton();
      if (attr.gpu_compatible()) {
        return ps->GetCUDAHostAllocator(0);
      } else {
        return cpu_allocator_;
      }
    } else {
      return gpu_allocator_;
    }
  }
};
```
从上面可以看出，`gpu_allocator_`和`cpu_allocator_`其实在`GPUDevice`构造时就已经传入准备好了。
tensorflow\core\common_runtime\gpu\gpu_device.cc
```CPP
Status BaseGPUDeviceFactory::CreateGPUDevice(const SessionOptions& options,
                                             const string& name, int gpu_id,
                                             BaseGPUDevice** out_device) {
                                                 ...
  ProcessState* process_state = ProcessState::singleton();
  *out_device = CreateGPUDevice(
      options, name, allocated_bytes, dev_locality, gpu_id,
      GetShortDeviceDescription(gpu_id, desc),
      process_state->GetGPUAllocator(options.config.gpu_options(), gpu_id,
                                     allocated_memory),
      process_state->GetCPUAllocator(numa_node));
...
```
而通过GPUDevice factory来创建GPUDevice，是在创建Session的时候会被执行的。
NewSession => DeviceFactory creates GPU device => ProcessState 创建gpu allocator

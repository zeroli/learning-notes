# 关于TF中的device
**Date: 11/21/2021**

```SH
tensorflow\core\common_runtime\device_factory.h
tensorflow\core\common_runtime\device_factory.cc
tensorflow\core\common_runtime\device.h
tensorflow\core\common_runtime\device.cc
tensorflow\core\common_runtime\device_mgr.h
tensorflow\core\common_runtime\device_mgr.cc
tensorflow\core\common_runtime\device_set.h
tensorflow\core\common_runtime\device_set.cc
```

下面的代码是用来注册不同设备的factory类：
```CPP
namespace dfactory {

template <class Factory>
class Registrar {
 public:
  // Multiple registrations for the same device type with different priorities
  // are allowed. The registration with the highest priority will be used.
  explicit Registrar(const string& device_type, int priority = 0) {
    DeviceFactory::Register(device_type, new Factory(), priority);
  }
};

}  // namespace dfactory

#define REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, ...) \
  INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory,   \
                                         __COUNTER__, ##__VA_ARGS__)

#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, \
                                               ctr, ...)                    \
  static ::tensorflow::dfactory::Registrar<device_factory>                  \
      INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr)(device_type,         \
                                                       ##__VA_ARGS__)

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr) ___##ctr##__object_
```
基本上就是提供宏来方便客户端代码，创建一个静态的类对象，从而在模块初始化时，自动注册，调用Registrar类的构造函数。

`DeviceFactory`类可以想见就是一个基于device_type字符串到DeviceFactory指针的map存储器。
但是这个类本身数据几乎时空的，但它是所有devicefactory的基类，虚函数需要重新实现，在每一个子类中。
```CPP
  virtual Status CreateDevices(const SessionOptions& options,
                               const string& name_prefix,
                               std::vector<Device*>* devices) = 0;
```

外界最终调用它的静态函数`AddDevices`来获取所有注册的设备工厂类可以创建的所有设备。
```CPP
  static Status AddDevices(const SessionOptions& options,
                           const string& name_prefix,
                           std::vector<Device*>* devices);
```
上面的`devices`就是输出，包含所有的设备对象。
```CPP
Status DeviceFactory::AddDevices(const SessionOptions& options,
                                 const string& name_prefix,
                                 std::vector<Device*>* devices) {
  // CPU first. A CPU device is required.
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered.  Did you link in threadpool_device?");
  }
  size_t init_size = devices->size();
  cpu_factory->CreateDevices(options, name_prefix, devices);
  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  // Then the rest (including GPU).
  mutex_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    auto factory = p.second.factory.get();
    if (factory != cpu_factory) {
      TF_RETURN_IF_ERROR(factory->CreateDevices(options, name_prefix, devices));
    }
  }

  return Status::OK();
}
```
从上面的实现代码中可以直到，CPU类型的设备一定要存在。

* GPU 设备工厂: `GPUDeviceFactory`
```SH
tensorflow\core\common_runtime\gpu\gpu_device.h
tensorflow\core\common_runtime\gpu\gpu_device.cc
tensorflow\core\common_runtime\gpu\gpu_device_factory.cc
```
DeviceFactory => BaseGPUDeviceFactory => GPUDeviceFactory
```CPP
REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory);
```
对应创建的设备子类： `GPUDevice`
`LocalDevice` => `BaseGPUDevice` => `GPUDevice`

* GPU兼容的CPU设备工厂: `GPUCompatibleCPUDeviceFactory`
```CPP
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 50);
```
对应创建的设备子类：`GPUCompatibleCPUDevice`
`LocalDevice` => `ThreadPoolDevice` => `GPUCompatibleCPUDevice`

* 线程池设备工厂，注册成CPU类型的工厂: `ThreadPoolDeviceFactory`
```SH
tensorflow\core\common_runtime\threadpool_device_factory.h
tensorflow\core\common_runtime\threadpool_device_factory.cc
```
```CPP
REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory);
```
对应创建的设备子类：`ThreadPoolDevice`
`LocalDevice` => `ThreadPoolDevice`

`LocalDevice`类定义在下面这个文件：
tensorflow\core\common_runtime\local_device.h

`LocalDevice`是继承于基类：`Device` (tensorflow\core\common_runtime\device.h)
*** 这个类需要稍后仔细研究下  ***

* 类`ThreadPoolDevice`，继承于`LocalDevice`
tensorflow\core\common_runtime\threadpool_device.h
tensorflow\core\common_runtime\threadpool_device.cc
来看下这个类实现的父类的一些接口函数：
```CPP
void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
    op_kernel->Compute(context);
  } else {
    op_kernel->Compute(context);
  }
}

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}

Status ThreadPoolDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   ProtoDebugString(tensor_proto));
  }
  *tensor = parsed;
  return Status::OK();
}
```
** TODO：**
1. 我们需要深入理解下Tracing的工作机制
2. 我们需要阅读Tensor类的代码
3. `OpKernel`类和`OpKernelContext`的运作机制
4. 谁调用到Device‘s Compute函数？？

题外话： 想学习下：StrCat的实现代码，下面的代码有时候很方便的：
`string name = strings::StrCat(name_prefix, "/cpu:", i);`


`ThreadPoolDeviceFactory`创建CPU devices过程：
```CPP
class ThreadPoolDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    // TODO(zhifengc/tucker): Figure out the number of available CPUs
    // and/or NUMA configuration.
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/cpu:", i);
      devices->push_back(new ThreadPoolDevice(
          options, name, Bytes(256 << 20), DeviceLocality(), cpu_allocator()));
    }

    return Status::OK();
  }
};
```
其中`cpu_allocator()`函数总是返回全局唯一个cpu allocator (tensorflow\core\framework\allocator.cc):
```CPP
Allocator* cpu_allocator() {
  static Allocator* cpu_alloc = MakeCpuAllocator();
  return cpu_alloc;
}
```

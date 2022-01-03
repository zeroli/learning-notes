#这个主题是关于tensorflow中cudnn handle的锁是否是全局锁
---------------------

## tensorflow's cudnn cudnnCreate(...)
那个在多线程下是只有唯一一个然后用mutex保护起来的？
还是多个session下，每个session一个，即使可能cudnn返回给我们的是相同的值，但是在不同的local，所以有不同的mutex来保护，也就没有global lock的问题？


cudnnAccess是在factory创建的，factory是全局的，只创建一次.

cuda_dnn.cc最后，注册了一个工厂函数，如何创建cudaSupport，全局性的.
```CPP
void initialize_cudnn() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
          cuda::kCudaPlatformId, gpu::kCuDnnPlugin, "cuDNN",
          [](internal::StreamExecutorInterface* parent) -> dnn::DnnSupport* {
            gpu::GpuExecutor* cuda_executor =
                dynamic_cast<gpu::GpuExecutor*>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the cuDNN "
                         << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            gpu::CudnnSupport* dnn = new gpu::CudnnSupport(cuda_executor);
            if (!dnn->Init().ok()) {
              // Note: Init() will log a more specific error.
              delete dnn;
              return nullptr;
            }
            return dnn;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuDNN factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kDnn, gpu::kCuDnnPlugin);
}

}  // namespace stream_executor

#pragma clang diagnostic pop

REGISTER_MODULE_INITIALIZER(register_cudnn,
                            { stream_executor::initialize_cudnn(); });

```
streamExecutor调用了convolve时候，它就需要dnnSupport:
parent_=> StreamExecutor
```CPP
Stream &Stream::ThenConvolveWithAlgorithm(
    const dnn::BatchDescriptor &input_descriptor,
    const DeviceMemory<double> &input_data,
    const dnn::FilterDescriptor &filter_descriptor,
    const DeviceMemory<double> &filter_data,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor, DeviceMemory<double> *output,
    ScratchAllocator *scratch_allocator,
    const dnn::AlgorithmConfig &algorithm_config,
    dnn::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(input_descriptor), PARAM(input_data),
            PARAM(filter_descriptor), PARAM(filter_data),
            PARAM(convolution_descriptor), PARAM(output_descriptor),
            PARAM(output), PARAM(algorithm_config));

  if (ok()) {
    if (dnn::DnnSupport *dnn = parent_->AsDnn()) {   ******
      DeviceMemory<uint8> scratch_memory;
      dnn::AlgorithmDesc algorithm_desc;
      auto status =
          dnn->PrepareForConvolution(
                 dnn::ConvolutionKind::FORWARD, this, input_descriptor,
                 input_data, filter_descriptor, filter_data, output_descriptor,
                 *output, convolution_descriptor, algorithm_config,
                 scratch_allocator, &algorithm_desc, &scratch_memory)
              .ok();
      if (status) {
        status = dnn->DoConvolve(
            this, input_descriptor, input_data, filter_descriptor, filter_data,
            convolution_descriptor, output_descriptor, output, algorithm_desc,
            &scratch_memory, output_profile_result);
      }
      if (!status && !output_profile_result) {
        SetError();
      }
    } else {
      SetErrorAndLogNoDnnSupport();
    }
  }
  return *this;
}
```
==>>>> tensorflow\stream_executor\stream_executor_pimpl.cc
std::unique_ptr<...> dnn_ => StreamExecutor's member data
```CPP
dnn::DnnSupport *StreamExecutor::AsDnn() {
  absl::MutexLock lock(&mu_);
  if (dnn_ != nullptr) {
    return dnn_.get();
  }

  dnn_.reset(implementation_->CreateDnn());
  return dnn_.get();
}
```
==> gpuExecutor::CreateDnn() at tensorflow\stream_executor\cuda\cuda_gpu_executor.cc:784
```CPP
dnn::DnnSupport* GpuExecutor::CreateDnn() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(cuda::kCudaPlatformId,
                                                       plugin_config_.dnn());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}
```
=>>> 调用到这里: gpu::CudnnSupport* dnn = new gpu::CudnnSupport(cuda_executor);
=>>> 调用Init(...)
```CPP
port::Status CudnnSupport::Init() {
  ScopedActivateExecutorContext context(parent_);
  cudnnHandle_t cudnn_handle = nullptr;
  const auto status = cudnnCreate(&cudnn_handle);  ******
  if (status == CUDNN_STATUS_SUCCESS) {
    CudnnVersion source_version(CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);

    CudnnVersion loaded_version;
    TF_RETURN_IF_ERROR(GetLoadedCudnnVersion(&loaded_version));
    if (!IsSourceCompatibleWithCudnnLibrary(source_version, loaded_version)) {
      const string error = absl::StrCat(
          "Loaded runtime CuDNN library: ", loaded_version.ToString(),
          " but source was compiled with: ", source_version.ToString(),
          ".  CuDNN library major and minor version needs to match or have "
          "higher minor version in case of CuDNN 7.0 or later version. If "
          "using a binary install, upgrade your CuDNN library.  If building "
          "from sources, make sure the library loaded at runtime is "
          "compatible "
          "with the version specified during compile configuration.");
      LOG(ERROR) << error;
      cudnnDestroy(cudnn_handle);
      return port::Status(port::error::INTERNAL, error);
    }

    cudnn_.reset(new CudnnAccess(cudnn_handle));  *******
    return port::Status::OK();
  }
  ```

=>>>> CudaAccess::GetHandle
```CPP
  CudnnHandle GetHandle(GpuExecutor* executor, Stream* stream) {
    auto lock = absl::make_unique<absl::MutexLock>(&mutex_);   /// ****** 锁住了
    mutex_.AssertHeld();
    gpu::ScopedActivateExecutorContext context(executor);
    CUstream cu_stream = stream ? AsGpuStreamValue(stream) : cudaStreamLegacy;
    const auto status = cudnnSetStream(handle_, cu_stream);
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "Failed to set cuDNN stream.";
    return CudnnHandle(std::move(context), std::move(lock), handle_);    /// ****** 锁住了
  }

class CudnnHandle {
 public:
  // Takes ownership of the executor context and the lock to access cuDNN
  // using handle.
  CudnnHandle(gpu::ScopedActivateExecutorContext context,
              std::unique_ptr<absl::MutexLock> lock, cudnnHandle_t handle)
      : context_(std::move(context)), lock_(std::move(lock)), handle_(handle) {}

  // Returns cuDNN handle. To be passed directly to cuDNN APIs, don't keep
  // a copy.
  cudnnHandle_t handle() const { return handle_; }

 private:
  gpu::ScopedActivateExecutorContext context_;
  std::unique_ptr<absl::MutexLock> lock_;
  cudnnHandle_t handle_;  // Not owned.
};
```
=>>>> cudnn.handle()拿到cudnn 返回的handle
```CPP
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  const auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnTransformTensor(   // *** 调用cudnn API 知道返回，然后释放cudnn handle
        cudnn.handle(), &scale, input_tensor_desc.handle(), input_data.opaque(),
        &beta, output_tensor_desc.handle(), output_data->opaque()));
    return port::Status::OK();
  }();
```


cudnSupport类成员：
```CPP
  // Provides access to the cuDNN handle.
  std::unique_ptr<class CudnnAccess> cudnn_;
```
在StreamExecutor销毁时，会销毁dnn_，然后cudnnSupport销毁也会销毁这个CudaAccess
```CPP
  ~CudnnAccess() {
    absl::MutexLock lock(&mutex_);
    cudnnDestroy(handle_);
  }
```

每次GetHandle(...)都会lock上面这个mutex，然后直到cudnn (by auto cudnn = cudnn_->GetHandle(parent_, stream)；)，goes out of scope，unique_ptr就会销毁，然后自动unlock。
在CudnnAccess销毁时， 就会destroy这个handle.

所以来说, StreamExecutor销毁，会最终destroy cudnn's handle.
```CPP
// Memoized DNN support object -- we only want to create this once when asked
// for an DNN interface.
std::unique_ptr<dnn::DnnSupport>  dnn_  GUARDED_BY(mu_);
```
故而这个StreamExecutor应该时全局性的，singleton类???? (**NO**)

StreamExecutor是如何创建出来的？
下面是callstack
>(gdb)
#0  stream_executor::StreamExecutor::StreamExecutor (this=0x45962d0, platform=<optimized out>, implementation=...) at tensorflow/stream_executor/stream_executor_pimpl.cc:148
#1  0x00007fffdb3dc6cb in absl::make_unique<stream_executor::StreamExecutor, stream_executor::gpu::CudaPlatform* const, std::unique_ptr<stream_executor::gpu::GpuExecutor, std::default_delete<stream_executor::gpu::GpuExecutor> > > () at external/com_google_absl/absl/memory/memory.h:168
#2  stream_executor::gpu::CudaPlatform::GetUncachedExecutor (this=0x7ef740, config=...) at tensorflow/stream_executor/cuda/cuda_platform.cc:174
#3  0x00007fffdb3db8ac in stream_executor::gpu::CudaPlatform::__lambda6::operator() (__closure=<optimized out>) at tensorflow/stream_executor/cuda/cuda_platform.cc:168
#4  std::_Function_handler<stream_executor::port::StatusOr<std::unique_ptr<stream_executor::StreamExecutor, std::default_delete<stream_executor::StreamExecutor> > >(), stream_executor::gpu::CudaPlatform::GetExecutor(const stream_executor::StreamExecutorConfig&)::__lambda6>::_M_invoke(const std::_Any_data &) (__func
tor=...) at /usr/lib/gcc/x86_64-redhat-linux/4.8.5/../../../../include/c++/4.8.5/functional:2057
#5  0x00007fffe01e7863 in std::function<stream_executor::port::StatusOr<std::unique_ptr<stream_executor::StreamExecutor, std::default_delete<stream_executor::StreamExecutor> > > ()>::operator()() const (this=0x7fffffffb380) at /usr/lib/gcc/x86_64-redhat-linux/4.8.5/../../../../include/c++/4.8.5/functional:2471
#6  stream_executor::ExecutorCache::GetOrCreate(stream_executor::StreamExecutorConfig const&, std::function<stream_executor::port::StatusOr<std::unique_ptr<stream_executor::StreamExecutor, std::default_delete<stream_executor::StreamExecutor> > > ()> const&) (this=this@entry=0x7ef750, config=..., factory=...) at ten
sorflow/stream_executor/executor_cache.cc:55
#7  0x00007fffdb3db950 in stream_executor::gpu::CudaPlatform::GetExecutor (this=0x7ef740, config=...) at tensorflow/stream_executor/cuda/cuda_platform.cc:168
#8  0x00007fffdb3dd205 in stream_executor::gpu::CudaPlatform::ExecutorForDevice (this=0x7ef740, ordinal=<optimized out>) at tensorflow/stream_executor/cuda/cuda_platform.cc:153
#9  0x00007fffdae72fc5 in tensorflow::GpuIdUtil::ExecutorForPlatformGpuId (platform_gpu_id=..., gpu_manager=0x7ef740) at ./tensorflow/core/common_runtime/gpu/gpu_id_utils.h:35
#10 tensorflow::(anonymous namespace)::GetPeerAccessMap (visible_gpu_order=std::vector of length 1, capacity 1 = {...}, platform=0x7ef740) at tensorflow/core/common_runtime/gpu/gpu_device.cc:1345
#11 tensorflow::BaseGPUDeviceFactory::GetInterconnectMaps (this=<optimized out>, visible_gpu_order=std::vector of length 1, capacity 1 = {...}, gpu_manager=0x7ef740, maps=0x7fffffffb670) at tensorflow/core/common_runtime/gpu/gpu_device.cc:1363
#12 0x00007fffdae7abc1 in tensorflow::BaseGPUDeviceFactory::CreateDevices (this=0x7b7830, options=..., name_prefix="/job:localhost/replica:0/task:0", devices=0x7fffffffba90) at tensorflow/core/common_runtime/gpu/gpu_device.cc:1176
#13 0x00007fffdaeba2fd in tensorflow::DeviceFactory::AddDevices (options=..., name_prefix="/job:localhost/replica:0/task:0", devices=devices@entry=0x7fffffffba90) at tensorflow/core/common_runtime/device_factory.cc:139
#14 0x00007fffe2be261d in tensorflow::DirectSessionFactory::NewSession (this=0x133caa0, options=..., out_session=0x7fffffffc388) at tensorflow/core/common_runtime/direct_session.cc:167
#15 0x00007fffdaf2c09e in tensorflow::NewSession (options=..., out_session=0x7fffffffc388) at tensorflow/core/common_runtime/session.cc:88

tensorflow\stream_executor\cuda\cuda_platform.cc
tensorflow\stream_executor\executor_cache.cc

StreamExecutor创建完之后会cache在executor_cache里面（executor_cache_）
```CPP
port::StatusOr<StreamExecutor*> CudaPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
CudaPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<GpuExecutor>(config.plugin_config));
  auto init_status = executor->Init(config.ordinal, config.device_options);
  if (!init_status.ok()) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for CUDA device ordinal %d: %s",
            config.ordinal, init_status.ToString()));
  }

  return std::move(executor);
}
```
每一个device有一个StreamExecutor????   (**YES**)
```CPP
port::StatusOr<StreamExecutor*> ExecutorCache::GetOrCreate(
    const StreamExecutorConfig& config,
    const std::function<ExecutorFactory>& factory) {
  // In the fast path case, the cache already has an entry and we can just
  // return after Get() which only takes a shared lock and not a unique lock.
  // If we need to create, we take a unique lock on cache_.
  auto fast_result = Get(config);
  if (fast_result.ok()) {
    return fast_result;
  }

  Entry* entry = nullptr;
  {
    absl::MutexLock lock{&mutex_};
    entry = &cache_[config.ordinal];
    // Release the map lock; the address of 'entry' is stable because
    // std::map guarantees reference stability.
  }

  // Acquire the per-Entry mutex without holding the map mutex. Initializing
  // an Executor may be expensive, so we want to allow concurrent
  // initialization of different entries.
  absl::MutexLock lock{&entry->configurations_mutex};
  for (const auto& iter : entry->configurations) {
    if (iter.first.plugin_config == config.plugin_config &&
        iter.first.device_options == config.device_options) {
      VLOG(2) << "hit in cache";
      return iter.second.get();
    }
  }

  VLOG(2) << "building executor";
  port::StatusOr<std::unique_ptr<StreamExecutor>> result = factory();
  if (!result.ok()) {
    VLOG(2) << "failed to get build executor: " << result.status();
    // If construction failed, leave the cache Entry around, but with a null
    // executor.
    return result.status();
  }
  entry->configurations.emplace_back(config, std::move(result.ValueOrDie()));
  return entry->configurations.back().second.get();
}
```
每个device会有一个StreamExecutor跟其对应，从而每个device上会有一个cudnnHandle
> [3.1.1.5. cudnnHandle_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnHandle_t)
cudnnHandle_t  is a pointer to an opaque structure holding the  cuDNN  library context. The  cuDNN  library context must be created using  [cudnnCreate()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate)  and the returned handle must be passed to all subsequent library function calls. The context should be destroyed at the end using  [cudnnDestroy()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroy). The context is associated with only one GPU device, the current device at the time of the call to  [cudnnCreate()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate). However, multiple contexts can be created on the same GPU device.

换句话说，session => StreamExecutor per devide => cudnn Handle per device，即使每次调用cudnn API都要获得lock，因为只有一个线程运行cudnn，情况还好。
但是如果多个RPCs同时运行在一个device上，因为StreamExecutor 是per device的，故而StreamExecutor将会被shared，进而它下面的cudnn Handle也是被shared，所以调用cudnn API之前，有多个线程会抢占那个handle lock。

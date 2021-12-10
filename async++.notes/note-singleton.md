# singleton模板类的学习

C++11的实现
=====
因为C++11标准支持线程安全的静态初始化，实现线程安全的singleton就比较简单和支持：
```CPP {.numberLines}
template<typename T>
class singleton {
public:
	static T& get_instance()
	{
		static T instance;
		return instance;
	}
};
```
但是上述实现有个limitation，就是无法构造带有参数的singleton对象。

不支持thread-safe静态初始化的编译器模拟版本：
======
```CPP {.numberLines}
template<typename T>
class singleton {
	std::mutex lock;
	std::atomic<bool> init_flag;
	typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type storage;

	static singleton instance;

	// Use a destructor instead of atexit() because the latter does not work
	// properly when the singleton is in a library that is unloaded.
	~singleton()
	{
		if (init_flag.load(std::memory_order_acquire))
			reinterpret_cast<T*>(&storage)->~T();
	}

public:
	static T& get_instance()
	{
		T* ptr = reinterpret_cast<T*>(&instance.storage);
		if (!instance.init_flag.load(std::memory_order_acquire)) {
			std::lock_guard<std::mutex> locked(instance.lock);
			if (!instance.init_flag.load(std::memory_order_relaxed)) {
				new(ptr) T;
				instance.init_flag.store(true, std::memory_order_release);
			}
		}
		return *ptr;
	}
};

template<typename T> singleton<T> singleton<T>::instance;
```
典型的基于double lock check机制实现的版本，但还是有些可以学习的地方：
- 采用另外一个atomic bool变量来去除一个mutex lock check，但使用的内存模式是`memory_order_acquire`；在mutex lock里面再用`memory_order_relaxed`内存模式进一步check；
- 采用aligned storage的方式提供对象存储空间，因此会使用placement new operator来构造对象；
- 这里可以直接在头文件中定义模板类的静态函数；
- 在singleton对象析构函数中，会去调用T类型的析构函数；这个析构函数会最终在程序退出时被调用；

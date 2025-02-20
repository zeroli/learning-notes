##
Introduction
===============
这个笔记是关于thrust-1.0代码的重新阅读。
我们首先从Thrust quick start pdf文档开始
- container， 容器
- algorithm， 算法
- functor, 算子
- iterator， 迭代器
- allocator， 分配器

### container， 容器
====================
thrust提供的容器很少，几乎只有一种`thrust::device_vector`,
对应的也提供了`thrust::host_vector`，对应STL的`std::vector<T>`。
`thrust::device_vector`和`thrust::host_device`都继承于`thrust::detail::vector_base`类
基类提供常用的构造/拷贝操作， clear/empty/resize/begin/end/front/back。
需要注意的是基类提供的API中，如果是否访问类的操作，定义为`__host__ __device__`，
也就是说子类`device_vector`是可以在CUDA kernel/device函数中被使用的。

`host_vector`提供的方法API就比较精简，全都是构造：
- 构造n个相同元素的vector，
- 拷贝构造/赋值，从相同元素类型的`host_vector`或者`device_vector`，
- 从其它类型的host_vector构造和拷贝构造，
- 从其它元素类型的std::vector构造和拷贝构造,
- 从一组输入迭代器构造。
- 同时提供operator ==，比较两个host_vector是否相同， host_vector和std::vector是否相同

`host_vector`定义在host_vector.h中，实现定义在host_vector.inl文件中， 这是一个比较好的声明和实现分离的代码组织方式。

### algorithm， 算法
====================
- transform算法
- reduction算法
- scan算法 (prefix sum， inclusive/exclusive scan)
- reordering算法
- sorting算法


### functor, 算子
====================


### iterator， 迭代器
====================
- constant iterator
- counting iterator
- transform iterator
- permutation iterator
- zip iterator


### allocator， 分配器
====================
`device_allocator`，继承于`device_new_alloc`

`device_ptr`针对raw pointer on device的简单封装
`device_reference`针对引用类型的封装，是对T&的一个简单抽象
```c++
typedef T                                 value_type;
typedef device_ptr<T>                     pointer;
typedef device_ptr<const T>               const_pointer;
typedef device_reference<T>               reference;
typedef device_reference<const T>         const_reference;
typedef std::size_t                       size_type;
typedef typename pointer::difference_type difference_type;
```
`device_ptr`和`device_reference`是什么关系呢？
自己写了一个`Ptr<T>`和`Ref<T>`，抽象pointer和reference的关系，可以参考代码文件`ptr_ref_example.cc`。

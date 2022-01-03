# aligned_alloc的学习

类aligned_array
=======
```CPP {.numberLines}
template<typename T, std::size_t Align = std::alignment_of<T>::value>
class aligned_array {
	std::size_t length;
	T* ptr;
```
模板类，功能类似于new T[N]，类对它进行了封装，提供如下接口：
- T& operator [](size_t)
- size_t size() const
- T* get() const
- explicit operator bool() const

主要的构造函数如下：
```CPP {.numberLines}
explicit aligned_array(std::size_t length)
    : length(length)
{
    ptr = static_cast<T*>(aligned_alloc(length * sizeof(T), Align));
    std::size_t i;
    LIBASYNC_TRY {
        for (i = 0; i < length; i++)
            new(ptr + i) T;
    } LIBASYNC_CATCH(...) {
        for (std::size_t j = 0; j < i; j++)
            ptr[i].~T();
        aligned_free(ptr);
        LIBASYNC_RETHROW();
    }
}
```
- 因为需要内存对齐，故而采用`aligned_alloc`先分配一段内存，aligned to `std::alignment_of(T)`，或用户提供的模板参数`Align`;
- 一个一个的调用的placement new操作，在一段内存上调用构造函数
- 构造函数有可能抛出异常，代码进行了处理：对已构造完成的对象调用析构函数（但不是reversed的调用），然后释放内存。这个过程其实就是new T[n]代码的实现逻辑。
- `align_alloc`在linux 平台上调用`posix_memaling`来实现，`align_free`简单调用`free`来释放。


**`aligned_array`不支持拷贝构造和拷贝赋值，支持移动构造和移动赋值**

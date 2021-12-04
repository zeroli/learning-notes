# stl alloc的学习和研究

文件列表：
- alloc.h
- stl_alloc.h

### `__malloc_alloc_template`类
在stl_alloc.h中有这样的一个类，它是个class template。它的template paramter是一个non-type integral，这样的设计就可以允许多个实例化类存在。主要愿意在于这个模板类有static数据变量`__malloc_alloc_oom_handler`存在，那是一个属于类的数据，不同实例化的类，可以拥有属于它们自己的不同的静态数据存在了。

```CPP {.numberLines}
template <int __inst>
class __malloc_alloc_template {
private:

  static void* _S_oom_malloc(size_t);
  static void* _S_oom_realloc(void*, size_t);

#ifndef __STL_STATIC_TEMPLATE_MEMBER_BUG
  static void (* __malloc_alloc_oom_handler)();
#endif
```
```CPP {.numberLines}
  static void* allocate(size_t __n)
  {
    void* __result = malloc(__n);
    if (0 == __result) __result = _S_oom_malloc(__n);
    return __result;
  }
```
分配失败时，调用OOM handler进行处理，处理方式如下：
```CPP {.numberLines}
template <int __inst>
void*
__malloc_alloc_template<__inst>::_S_oom_malloc(size_t __n)
{
    void (* __my_malloc_handler)();
    void* __result;

    for (;;) {
        __my_malloc_handler = __malloc_alloc_oom_handler;
        if (0 == __my_malloc_handler) { __THROW_BAD_ALLOC; }
        (*__my_malloc_handler)();  // 调用oom handler进行处理，之后再重新malloc
        __result = malloc(__n);
        if (__result) return(__result);
    }
}
```
看下它自己typedef了一个malloc_alloc版本出来：
`typedef __malloc_alloc_template<0> malloc_alloc;`

### `simple_alloc`类
这个类定义为一个模版类，模板参数是数据类型和内存分配类型。里面的函数都定义为static。
```CPP {.numberLines}
template<class _Tp, class _Alloc>
class simple_alloc {

public:
    static _Tp* allocate(size_t __n)
      { return 0 == __n ? 0 : (_Tp*) _Alloc::allocate(__n * sizeof (_Tp)); }
    static _Tp* allocate(void)
      { return (_Tp*) _Alloc::allocate(sizeof (_Tp)); }
    static void deallocate(_Tp* __p, size_t __n)
      { if (0 != __n) _Alloc::deallocate(__p, __n * sizeof (_Tp)); }
    static void deallocate(_Tp* __p)
      { _Alloc::deallocate(__p, sizeof (_Tp)); }
};
```
`allocate`和`deallocate`静态函数，要求模板参数`_Alloc`提供静态函数`allocate`和`deallocate`，但是参数是自己与字节，因为它不跟具体类型绑定。

如果一个allocator类不以具体的类型进行实例化，那么它的分配最好是基于字节的，否则参数是个数。

以上内容都是基于malloc/free定义的简单内存分配器。

----

sgi-stl还定义了一个基于bin list的定制版的allocator：`__default_alloc_template`
### `__default_alloc_template`类


### `allocator`类
这个类是兼容于std c++标准的内存分配器，采用模板类来实现。
```CPP {.numberLines}
template <class _Tp>
class allocator {
  typedef alloc _Alloc;          // The underlying allocator.
public:
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;
  typedef _Tp*       pointer;
  typedef const _Tp* const_pointer;
  typedef _Tp&       reference;
  typedef const _Tp& const_reference;
  typedef _Tp        value_type;

  template <class _Tp1> struct rebind {
    typedef allocator<_Tp1> other;
  };
```
- 模版类，模板参数是数据类型_Tp
- 采用`alloc`进行底层分配，`alloc`要么是上面的简单内存分配器，要么是基于bin list的定制版内存分配器，基于编译选项来控制。
- std c++标准要求一些内嵌类型必须定义: `size_type`, `different_type`, `pointer`, `const_pointer`, `reference`, `const_reference`, `value_type`。
- 还有一个比较特别的要求是需要定义struct rebind类，如果将当前allocator rebind到基于另外一种数据类型的内存分配。这里采用相同模版类，实例化不同的数据类型。

看下它2个最终要的接口函数定义: `allocate`和`deallocate`
```CPP {.numberLines}
  // __n is permitted to be 0.  The C++ standard says nothing about what
  // the return value is when __n == 0.
  _Tp* allocate(size_type __n, const void* = 0) {
    return __n != 0 ? static_cast<_Tp*>(_Alloc::allocate(__n * sizeof(_Tp)))
                    : 0;
  }

  // __p is not permitted to be a null pointer.
  void deallocate(pointer __p, size_type __n)
    { _Alloc::deallocate(__p, __n * sizeof(_Tp)); }
```
都是直接转调用到underlying的allocator类静态函数。

另外2个接口函数定义，构造和销毁。
```CPP {.numberLines}
  void construct(pointer __p, const _Tp& __val) { new(__p) _Tp(__val); }
  void destroy(pointer __p) { __p->~_Tp(); }
```
- 在提供的指针`_p`指向的内存空间上直接进行对象构造，placement new
- 提供类型指针`_p`，显式调用它的析构函数

通常标准兼容的allocator还需要显式特化基于void类型的allocator，以免客户代码隐式实例化这样的类，然后错误信息比较含糊。
```CPP {.numberLines}
template<>
class allocator<void> {
public:
  typedef size_t      size_type;
  typedef ptrdiff_t   difference_type;
  typedef void*       pointer;
  typedef const void* const_pointer;
  typedef void        value_type;

  template <class _Tp1> struct rebind {
    typedef allocator<_Tp1> other;
  };
};
```

标准还要求定义allocator的比较操作符：
```CPP {.numberLines}
template <class _T1, class _T2>
inline bool operator==(const allocator<_T1>&, const allocator<_T2>&)
{
  return true;
}

template <class _T1, class _T2>
inline bool operator!=(const allocator<_T1>&, const allocator<_T2>&)
{
  return false;
}
```
这样在STL container进行swap时，判断是否可以进行安全的swap。因为对于基于不同allocator的container，如果allocator不相等（不能安全swap），那么swap container是不安全的。上面的allocator相等比较，对于任何2种不同类型，都判定为相等， 因为allocator被定义为statless的（不绑定任何数据成员）。

还有一个类就是`_Alloc_traits`，需要好好学习下它的设计。
### `_Alloc_traits`类

# STL allocator/allocator traits

link: https://zhuanlan.zhihu.com/p/354191253

`std::allocator`和`__gnu_cxx::__alloc_traits`

`__gnu_cxx::new_allocator`
====
C++默认的内存分配器`std::allocator`，继承至`__gnu_cxx::new_allocator`，而后者主要完成2个任务：
- 分配对象内存、初始化对象；
- 析构对象、释放对象内存；

`__gnu_cxx::new_allocator`是一个空类（所有函数都是静态的），4个接口函数：
- `allocate`函数，用于分配内存；
- `construct`函数，调用已分配内存对象的构造函数；
- `destroy`函数，调用析构函数；
- `deallocate`函数，用于释放内存；

```c++
template <typename _Tp>
  class new_allocator
  {
  public:
    typedef size_t      size_type;
    typedef ptrdiff_t   difference_type;
    typedef _Tp*        pointer;
    typedef const _Tp*  const_pointer;
    typedef _Tp &       reference;
    typedef const _Tp & const_reference;
    typedef _Tp         value_type;

    template <typename _Tp1>
    struct rebind
    {
      typedef new_allocator<_Tp1> other;
    };

    new_allocator() _GLIBCXX_USE_NOEXCEPT {}
    new_allocator(const new_allocator &) noexcept {}
    template <typename _Tp1>
    new_allocator(const new_allocator<_Tp1> &) noexcept {}
    ~new_allocator() noexcept {}

    pointer allocate(size_type __n, const void * = static_cast<const void *>(0));

    void deallocate(pointer __p, size_type);

    size_type max_size() const noexcept;

    template <typename _Up, typename... _Args>
    void construct(_Up *__p, _Args &&...__args)
                    noexcept(noexcept(::new ((void *)__p)_Up(std::forward<_Args>(__args)...)));
    template <typename _Up>
    void destroy(_Up *__p) noexcept(noexcept(__p->~_Up()));
    //...
  };
```
- `allocate`函数：
```c++
pointer allocate(size_type __n, const void * = static_cast<const void *>(0))
    {
      if (__n > this->max_size())
        std::__throw_bad_alloc();

#if __cpp_aligned_new
      if (alignof(_Tp) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
      {
        std::align_val_t __al = std::align_val_t(alignof(_Tp));
        return static_cast<_Tp *>(::operator new(__n * sizeof(_Tp), __al));
      }
#endif
      return static_cast<_Tp *>(::operator new(__n * sizeof(_Tp)));
    }
```
- `deallocate`函数：
```c++
void deallocate(pointer __p, size_type)
    {
#if __cpp_aligned_new
      if (alignof(_Tp) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
      {
        ::operator delete(__p, std::align_val_t(alignof(_Tp)));
        return;
      }
#endif
      ::operator delete(__p);
    }
```

- `construct`函数：(函数模板，variadic template实现，支持任何参数构造)
```c++
template <typename _Up, typename... _Args>
    void construct(_Up *__p, _Args &&...__args)
                    noexcept(noexcept(::new ((void *)__p)_Up(std::forward<_Args>(__args)...)))
    {
        // 表示在 地址 _p 上调用对象 _Up的构造函数
        // 其中，__args是构造函数的参数
        ::new ((void *)__p) _Up(std::forward<_Args>(__args)...);
    }
```

- `destroy`函数：
```c++
template <typename _Up>
    void destroy(_Up *__p) noexcept(noexcept(__p->~_Up()))
    {
      __p->~_Up();
    }
```

`std::allocator`
===
`std::allocator`类继承至`__gnu_cxx::new_allocator`;
```c++
template<typename _Tp>
using __allocator_base = __gnu_cxx::new_allocator<_Tp>;

template <typename _Tp>
  class allocator : public __allocator_base<_Tp>
  {
  public:
    typedef size_t      size_type;
    typedef ptrdiff_t   difference_type;
    typedef _Tp*        pointer;
    typedef const _Tp*  const_pointer;
    typedef _Tp&        reference;
    typedef const _Tp&  const_reference;
    typedef _Tp         value_type;

    template <typename _Tp1>
    struct rebind
    {
      typedef allocator<_Tp1> other;
    };
...
```

rebind
===
在__gnu_cxx::new_allocator、std::allocator中都有一个rebind函数，其主要作用：获得类型_Tp1的内存分配器allocator<_Tp1>。
```c++
template <typename _Tp1>
    struct rebind
    {
      typedef allocator<_Tp1> other;
    };
```
譬如std::list中的代码用法如下：
```c++
template<typename _Tp, typename _Alloc>
    class _List_base
    {
    protected:
      // 用于分配 _Tp 类型的内存分配器: _Tp_alloc_type
      // _Tp_alloc_type 实际上就是 std::allocator
      typedef typename __gnu_cxx::__alloc_traits<_Alloc>::template rebind<_Tp>::other _Tp_alloc_type;
      // 用于分配 List_node<_Tp> 类型的内存分配器：_Node_alloc_type
      typedef typename _Tp_alloc_traits::template rebind<_List_node<_Tp> >::other _Node_alloc_type;
      //...
    };

    template<typename _Tp, typename _Alloc = std::allocator<_Tp> >
    class list : protected _List_base<_Tp, _Alloc>
    {
    protected:
      typedef _List_node<_Tp>                _Node;
      //...
    };
```

`std::__allocator_traits__base`
===
```c++
__gnu_cxx::__alloc_traits 继承自 std::allocator_traits
std::allocator_traits     继承自 std::__allocator_traits_base
```
`std::__allocator_traits_base`是个空类，非模板类，但是它所有成员函数或者嵌套类型都是模板；
```c++
struct __allocator_traits_base
  {
    template <typename _Tp,
              typename _Up,
              typename = void>
    struct __rebind : __replace_first_arg<_Tp, _Up>
    { };

    // __rebind 特化版本：当分配器 _Tp 有成员函数 rebind 时调用此特化版本
    template <typename _Tp, typename _Up>
    struct __rebind<_Tp,
                    _Up,
                    __void_t<typename _Tp::template rebind<_Up>::other>>
    {
      using type = typename _Tp::template rebind<_Up>::other;
    };

  protected:
    template <typename _Tp> using __pointer     = typename _Tp::pointer;
    template <typename _Tp> using __c_pointer   = typename _Tp::const_pointer;
    template <typename _Tp> using __v_pointer   = typename _Tp::void_pointer;
    template <typename _Tp> using __cv_pointer  = typename _Tp::const_void_pointer;
    template <typename _Tp> using __pocca = typename _Tp::propagate_on_container_copy_assignment;
    template <typename _Tp> using __pocma = typename _Tp::propagate_on_container_move_assignment;
    template <typename _Tp> using __pocs  = typename _Tp::propagate_on_container_swap;
    template <typename _Tp> using __equal = typename _Tp::is_always_equal;
  };
```
**__rebind类型**
* 当传入的内存分配器类型_Tp，实现了rebind成员函数时，比如std::allocator，那么就调用__rebind的特化版本；
`__allocator_traits_base::__rebind<std::allocator<int>, Node<int>>::type`
=>
`std::allocator<Node<int>>`
* 否则，就调用__rebind普通版本，继承自`__replace_first_arg`；
```c++
 template <typename _Tp, typename _Up>
    struct __replace_first_arg
    { };

    // _Template 是个类模板
    template <template <typename, typename...> class _Template,
              typename _Up,
              typename _Tp,
              typename... _Types>
    struct __replace_first_arg<_Template<_Tp, _Types...>, _Up>
    {
      using type = _Template<_Up, _Types...>;
    };
```
那么__rebind<xxx>::type将是一个`__Template<xxxx>`类型；其实也就是用户自定义的allocator类型，这个类型没有定义rebind；

`__alloc_rebind`
===
全局函数`__alloc_rebind`，是`std::__alloator_traits_base`的wrapper;
```c++
template <typename _Alloc, typename _Up>
 using __alloc_rebind = typename __allocator_traits_base::template __rebind<_Alloc, _Up>::type;
```

`std::allocator_traits`
===
继承于`std::__allocator_traits_base`类，用于获取allocator的各个属性；
当`_Alloc`是`std::allocator`时，std::allocator_traits有一个特化版本：
```c++
template <typename _Tp>
  struct allocator_traits<allocator<_Tp>>
```
**是针对值类型`_Tp`的模板参数，而不是以allocator作为模板参数；**
```c++
template <typename _Tp>
  struct allocator_traits<allocator<_Tp>>
  {
    using allocator_type = allocator<_Tp>;  // 分配器类型
    using value_type = _Tp;                 // 待分配内存的对象类型
    using pointer = _Tp *;                 // 对象指针
    using const_pointer = const _Tp *;
    //...  using
    using is_always_equal = true_type;

    // 使用allocator为_Up分配内存
    template <typename _Up>
    using rebind_alloc = allocator<_Up>;

    template <typename _Up>
    using rebind_traits = allocator_traits<allocator<_Up>>;

    // 下面是 std::allocator<_Tp> 成员函数的 wrapper
    static pointer allocate(allocator_type &__a, size_type __n)
    {
      return __a.allocate(__n);
    }

    static pointer allocate(allocator_type &__a, size_type __n, const_void_pointer __hint)
    {
      return __a.allocate(__n, __hint);
    }

    static void deallocate(allocator_type &__a, pointer __p, size_type __n)
    {
      __a.deallocate(__p, __n);
    }

    template <typename _Up, typename... _Args>
    static void construct(allocator_type &__a, _Up *__p, _Args &&...__args)
                noexcept(noexcept(__a.construct(__p, std::forward<_Args>(__args)...)))
    {
      __a.construct(__p, std::forward<_Args>(__args)...);
    }

    template <typename _Up>
    static void destroy(allocator_type &__a, _Up *__p) noexcept(noexcept(__a.destroy(__p)))
    {
      __a.destroy(__p);
    }

    static size_type max_size(const allocator_type &__a) noexcept
    {
      return __a.max_size();
    }
  };
```

`gnu_cxx::alloc_traits`
===
__gnu_cxx::__alloc_traits类，也大都是std::allocator_traits的wrapper;
不过它对non标准类型的指针进行了函数重载；
```c++
// 当 _Ptr 不是个标准指针，但是 _Ptr 和 value_type* 相同
    // __is_custom_pointer 才是 true，即 _Ptr 是个自定义指针
    // 即 _Ptr 可转换为 pointer
    template <typename _Ptr>
    using __is_custom_pointer
        = std::__and_<std::is_same<pointer, _Ptr>, std::__not_<std::is_pointer<_Ptr>>>;
```
```c++
// overload construct for non-standard pointer types
    // 重载非标准类型的指针，调用构造函数
    template <typename _Ptr, typename... _Args>
    static typename std::enable_if<__is_custom_pointer<_Ptr>::value>::type
    construct(_Alloc &__a, _Ptr __p, _Args &&...__args) noexcept(...) // 省略了noexcept中的表达式
    {
      // 使用分配器 __a , 在地址 __p 调用构造函数
      _Base_type::construct(__a,
                            std::__to_address(__p),
                            std::forward<_Args>(__args)...);
    }

    // overload destroy for non-standard pointer types
    // 重载非标准类型指针，调用析构函数
    template <typename _Ptr>
    static typename std::enable_if<__is_custom_pointer<_Ptr>::value>::type
    destroy(_Alloc &__a, _Ptr __p) noexcept(...)
    {
      _Base_type::destroy(__a, std::__to_address(__p));
    }

    /*** 对于标准的指针，会直接调用父类的constuct、destroy ***/

    // wrapper
    template <typename _Tp>
    struct rebind
    {
      typedef typename _Base_type::template rebind_alloc<_Tp> other;
    };
```

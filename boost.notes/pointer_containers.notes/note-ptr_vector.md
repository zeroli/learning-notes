# boost::ptr_vector class的学习

example
=====
```c++ {.numberLines}
#include <boost/ptr_container/ptr_vector.hpp>
#include <iostream>

int main()
{
  boost::ptr_vector<int> v;
  v.push_back(new int{1});
  v.push_back(new int{2});
  std::cout << v.back() << '\n';
}
```
- ptr_vector操作的是heap allocated的指针对象；
- `back()`返回的是指针de-reference后的对象；

```c++ {.numberLines}
    template
    <
        class T,
        class CloneAllocator = heap_clone_allocator,
        class Allocator      = void
    >
    class ptr_vector : public
        ptr_sequence_adapter< T,
            std::vector<
                typename ptr_container_detail::void_ptr<T>::type,
                typename boost::mpl::if_<boost::is_same<Allocator, void>,
                    std::allocator<typename ptr_container_detail::void_ptr<T>::type>, Allocator>::type
            >,
            CloneAllocator >
```

- 类`ptr_vector`继承于`ptr_sequence_adapter`模板类；
- adapter类以`std::vector`类底层的类型；
- 如果`Allocator`为`void`，然后vector的allocator便是system的allocator；
-
-
`heap_clone_allocator`
file: ptr_container\clone_allocator.hpp
======
```c++ {.numberLines}
    struct heap_clone_allocator
    {
        template< class U >
        static U* allocate_clone( const U& r )
        {
            return new_clone( r );
        }

        template< class U >
        static void deallocate_clone( const U* r )
        {
            delete_clone( r );
        }

    };
    template< class T >
    inline T* new_clone( const T& r )
    {
        //
        // @remark: if you get a compile-error here,
        //          it is most likely because you did not
        //          define new_clone( const T& ) in the namespace
        //          of T.
        //
        T* res = new T( r );
        BOOST_ASSERT( typeid(r) == typeid(*res) &&
                      "Default new_clone() sliced object!" );
        return res;
    }

    template< class T >
    inline void delete_clone( const T* r )
    {
        checked_delete( r );
    }
```
file: core\checked_delete.hpp
```c++ {.numberLines}
template<class T> inline void checked_delete(T * x) BOOST_NOEXCEPT
{
    // intentionally complex - simplification causes regressions
    typedef char type_must_be_complete[ sizeof(T)? 1: -1 ];
    (void) sizeof(type_must_be_complete);
    delete x;
}
```

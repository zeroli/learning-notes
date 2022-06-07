# check class has member (data or function, or type)

## check if one class has specific function/data
```c++
template <typename Archive, typename T>
struct has_member_serialize {
    typedef typename std::remove_reference<T>::type T2;
    typedef typename std::remove_cv<T2>::type T3;
    template <typename V, V>
    struct dummy {
    };

    template <typename U>
    static char test(dummy<void (U::*)(Archive&), &U::serialize>*);
    template <typename U>
    static int test(...);

    static const bool value = sizeof(test<T3>(0)) == sizeof(char);
};
```
* if it has, then first test will be instantialized, and return `char`

## detect member data (TYPE x)
```c++
    template <typename U>
    static char test(dummy<TYPE U::*, &U::X>*);
    template <typename U>
    static int test(...);
```
* if it has `X`, then first test will be instantialized, and return `char`

## detect member type
use `typename U::TYPE *` as argument type, in above example code

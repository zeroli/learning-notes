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

检测类是否有begin/end
1. 给出begin函数类型，强制转换想要的，然后跟期待的类型进行比较，一切都成功，就可以判断类存在想要的签名的函数
```c++
    template <typename C>
    static char test_begin(typename std::enable_if<
                           std::is_same<decltype(static_cast<typename C::const_iterator (C::*)() const>(&C::begin)),
                                        typename C::const_iterator (C::*)() const>::value,
                           void>::type*);
    template <typename C>
    static int test_begin(...);
```
2. BOOST hash库中的策略
```c++
template<class T, class It>
    integral_constant< bool, !is_same<typename remove_cv<T>::type, typename std::iterator_traits<It>::value_type>::value >
        is_range_check( It first, It last );

template<class T> decltype( is_range_check<T>( declval<T const&>().begin(), declval<T const&>().end() ) ) is_range_( int );
template<class T> false_type is_range_( ... );
```
`declval<T const&>().begin()`返回的类型，转入一个间接函数作为参数: `is_range_check`, 通过这个间接函数返回值来判断一切是否成功。
通过判断迭代器类型的内嵌value类型是否与T相同，来判断begin/end存在而且返回想要的迭代器。

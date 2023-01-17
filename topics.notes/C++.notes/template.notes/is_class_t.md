```c++
namespace detail {
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

template <typename T>
bool_constant<not std::is_union<T>::value>
is_class(int T::*);  //任何类都可以有成员指针，哪怕那个类是空类

template <typename T>
std::false_type is_class(...);
}

template <typename T>
using my_is_class = decltype(detail::is_class<T>(nullptr));

struct Foo { };
static_assert(my_is_class<Foo>::value);
static_assert(not my_is_class<int>::value);
```

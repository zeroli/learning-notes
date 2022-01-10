# class template SFINAE

```c++ {.numberLines}
#include <type_traits>

template <typename T, typename Enable = void>
class foo;

template <typename T>
class foo<T, typename std::enable_if<std::is_integral<T>::value>::type>
{ };

template <typename T>
class foo<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
{ };
```

Requires c++11 or newer.
INTENT
Conditionally instantiate a class template depending on the template arguments.

DESCRIPTION
We provide two partial specializations of the foo class template:

The template on lines 6–8 will only be instantiated when T is an integral type.
The template on lines 10–12 will only be instantiated when T is a floating point type.
This allows us to provide different implementations of the foo class depending on the template arguments it is instantiated with.

We have used std::enable_if on line 7 and line 11 to force instantiation to succeed only for the appropriate template arguments. This relies on Substitution Failure Is Not An Error (SFINAE), which states that failing to instantiate a template with some particular template arguments does not result in an error and simply discards that instantiation.

If you want to simply prevent a template from being instantiated for certain template arguments, consider using static_assert instead.

C++.INSIGHTS的结果：

```c++ {.numberLines}
#include <iostream>

#include <type_traits>
using namespace std;

template <typename T, typename Enable = void>
class foo;

/* First instantiated from: insights.cpp:19 */
#ifdef INSIGHTS_USE_TEMPLATE
template<>
class foo<int, void>
{
  public:
  // inline constexpr foo() noexcept = default;
};

#endif


/* First instantiated from: insights.cpp:20 */
#ifdef INSIGHTS_USE_TEMPLATE
template<>
class foo<double, void>
{
  public:
  // inline constexpr foo() noexcept = default;
};

#endif

template<typename T>
class foo<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
};


template<typename T>
class foo<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
{
};


int main()
{
  foo<int> fi = foo<int, void>();
  foo<double> fd = foo<double, void>();
  return 0;
}

```

- 模板类的主模板仅仅是申明，默认参数定义为`void`；
- 通过模板偏特化来实现if/else编译期决议到不同类的定义；
- 自然需要借助`std::enable_if`；
- 偏特化时，当然也可以根据类型进行表达式组合来判断适合表达式合法来决议是否可以选择这个偏特化版本，可以不借助`std::enable_if`；
    - 比如std::void_t<....>来判断...里面的类型是否合法；

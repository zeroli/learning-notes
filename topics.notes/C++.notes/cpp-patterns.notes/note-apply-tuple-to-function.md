# Apply tuple to function

```c++ {.numberLines}
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

template<typename F, typename Tuple, size_t ...S >
decltype(auto) apply_tuple_impl(F&& fn, Tuple&& t, std::index_sequence<S...>)
{
  return std::forward<F>(fn)(std::get<S>(std::forward<Tuple>(t))...);
}

template<typename F, typename Tuple>
decltype(auto) apply_from_tuple(F&& fn, Tuple&& t)
{
  std::size_t constexpr tSize
    = std::tuple_size<typename std::remove_reference<Tuple>::type>::value;

  return apply_tuple_impl(std::forward<F>(fn),
                          std::forward<Tuple>(t),
                          std::make_index_sequence<tSize>());
}

int do_sum(int a, int b)
{
  return a + b;
}

int main()
{
  int sum = apply_from_tuple(do_sum, std::make_tuple(10, 20));
}
```

Requires c++14 or newer.
INTENT
Unpack a tuple as the arguments of a function.

DESCRIPTION
The apply_from_tuple function template on lines 12–21 returns the result of applying the function fn to the values stored in the std::tuple t. On lines 15–16, we store the size of t in tSize, which is declared constexpr so that it can be evaluated at compile-time. On lines 18–20, we call apply_tuple_impl passing t, fn and an std::index_sequence which carries a parameter pack containing a sequence of integers from 0 to tSize - 1.

The apply_tuple_impl function template on lines 6–10 returns the result of applying the function fn using event element of the tuple t as arguments (on line 9). To do this, we expand the parameter pack carried by the std::index_sequence and apply std::get to the tuple for each integer in the sequence. This way, all the elements of t are expanded and passed to the function.

Note: a std::apply function has been proposed as part of the Library Fundamentals TS for future standardization.

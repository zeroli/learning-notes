```c++
namespace detail {
std::true_type is_null_ptr(std::nullptr_t);
std::false_type is_null_ptr(...);
}

template <typename T>
using is_nullptr = decltype(detail::is_null_ptr(std::declval<T>()));


static_assert(is_nullptr<std::nullptr_t>::value);
static_assert(not is_nullptr<int>::value);

```
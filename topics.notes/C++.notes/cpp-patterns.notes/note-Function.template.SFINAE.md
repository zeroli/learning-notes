# Function template SFINAE

```c++ {.numberLines}
#include <type_traits>
#include <limits>
#include <cmath>

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
  equal(T lhs, T rhs)
{
  return lhs == rhs;
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
  equal(T lhs, T rhs)
{
  return std::abs(lhs - rhs) < 0.0001;
}
```

Requires c++11 or newer.
INTENT
**Conditionally instantiate a function template depending on the template arguments.**

DESCRIPTION
We provide two implementations of the equal function template:

The template on lines 5–10 will only be instantiated when T is an integral type.
The template on lines 12–17 will only be instantiated when T is a floating point type.
We have used std::enable_if on line 6 and line 13 to force instantiation to succeed only for the appropriate template arguments. This relies on Substitution Failure Is Not An Error (SFINAE), which states that failing to instantiate a template with some particular template arguments does not result in an error and simply discards that instantiation.

The second template argument of std::enable_if — in this case, bool — is what the full std::enable_if<...>::type evaluates to when the first template argument is true. This means that the return type of equal will be bool.

If you want to simply prevent a template from being instantiated for certain template arguments, consider using static_assert instead.

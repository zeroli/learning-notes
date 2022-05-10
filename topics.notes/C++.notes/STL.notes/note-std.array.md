# std::array

Defined in header <array>
template<
    class T,
    std::size_t N
> struct array;
(since C++11)
std::array is a container that encapsulates fixed size arrays.

This container is an aggregate type with the same semantics as a struct holding a C-style array T[N] as its only non-static data member. Unlike a C-style array, it doesn't decay to T* automatically. As an aggregate type, it can be initialized with aggregate-initialization given at most N initializers that are convertible to T: std::array<int, 3> a = {1,2,3};.

- std::array并没有定义任何构造函数
- std::array也没有定义析构函数
- std::array没有定义拷贝和赋值操作符
  - 因为它是一种对C-STYLE数组的聚合类型，拷贝/赋值操作都是C++编译器隐式定义的；故而可以实现数组逐元素拷贝

```c++ {.numberLines}
#include <array>
#include <experimental/dynarray>
void compile_time(std::array<int, 3> arr)
{ }
void run_time(std::experimental::dynarray<int> arr)
{ }
int main()
{
  std::array<int, 3> arr = {4, 8, 15};
  compile_time(arr);
  compile_time({16, 23, 42});
  std::experimental::dynarray<int> dynarr = {1, 2, 3};
  run_time(dynarr);
  run_time({1, 2, 3, 4, 5});
}
```

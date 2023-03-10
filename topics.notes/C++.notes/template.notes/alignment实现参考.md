# 摘抄自thrust库的实现

```c++
namespace alignment_of_detail {


  template <typename T>
  class alignment_of_impl;

  template <typename T, std::size_t size_diff>
  struct helper
  {
    static const std::size_t value = size_diff;
  };

// 如果提供的T是个空类，没有大小，那么接下来强制加一个字节
  template <typename T>
  class helper<T, 0>
  {
  public:
    static const std::size_t value = alignment_of_impl<T>::value;
  };

  template <typename T>
  class alignment_of_impl
  {
  private:
    struct big
    {
      T    x;
      char c;
    };

  public:
    static const std::size_t value = helper<big, sizeof(big) - sizeof(T)>::value;
  };


}    // end alignment_of_detail

// std::pair<double, int>, alignment = 8, 后面添加1个字节, sizeof(big) = 24, sizeof(T) = 16
// std::pair<int, double>, alignment = 8, 后面添加1个字节, sizeof(big) = 24, sizeof(T) = 16
// alignment<char> = 1
// alignment<short> = 2
// alignment<int> = 4
// alignment<double> = 8
template <typename T>
struct alignment_of
    : alignment_of_detail::alignment_of_impl<T>
{
};
```
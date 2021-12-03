# bit_cast模板函数

tensorflow\core\lib\core\casts.h

> // bit_cast<Dest,Source> is a template function that implements the
> // equivalent of "*reinterpret_cast<Dest*>(&source)".  We need this in
> // very low-level functions like the protobuf library and fast math
> // support.
> //
> //   float f = 3.14159265358979;
> //   int i = bit_cast<int32>(f);
> //   // i = 0x40490fdb
> //
> // The classical address-casting method is:
> //
> //   // WRONG
> //   float f = 3.14159265358979;            // WRONG
> //   int i = * reinterpret_cast<int*>(&f);  // WRONG
> ...

```cpp
template <class Dest, class Source>
inline Dest bit_cast(const Source& source) {
  static_assert(sizeof(Dest) == sizeof(Source), "Sizes do not match");

  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}
```
单元测试程序比较清楚的演示它的用法。
tensorflow\core\lib\core\bit_cast_test.cc
基本上说int和浮点数之间的reinterpret_cast的基于地址转换都是undefined behavior，无法保证转来转去后底层内存字节bits保持不变，因为编译器会做些优化。

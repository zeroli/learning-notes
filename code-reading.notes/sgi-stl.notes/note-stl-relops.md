# stl-relops的学习

文件列表：
- stl_relops.h

这个文件定义了2个类型的relation operations函数
```CPP {.numberLines}
template <class _Tp>
inline bool operator!=(const _Tp& __x, const _Tp& __y) {
  return !(__x == __y);
}

template <class _Tp>
inline bool operator>(const _Tp& __x, const _Tp& __y) {
  return __y < __x;
}

template <class _Tp>
inline bool operator<=(const _Tp& __x, const _Tp& __y) {
  return !(__y < __x);
}

template <class _Tp>
inline bool operator>=(const _Tp& __x, const _Tp& __y) {
  return !(__x < __y);
}
```
- 基于相等`==`可以实现`!=`操作
- 基于小于`<`可以实现其它的相关运算符: `>`, `<=`, `>=`
  - `x > y` => `y < x`
  - `x <= y` => `!(x > y)` => `!(y < x)`
  - `x >= y` => `!(x < y)`

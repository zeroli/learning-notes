# fmtlib introduction

C++格式化输出库fmtlib
====

* motivation
====
- `printf`: 不安全
`printf("%.f", 0.);`

- C++的流式I/O： `cout`做到了类型安全，也做到了拓展性，但使用起来比较麻烦，而且效率不高；
```c++
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << 0.;
}
```

现代化的格式化库的样子
1. 方便使用；
2. 用户拓展性；
3. 安全性；
4. 性能I

python里面的2种格式化字符串的办法：
```python
name = 'Steven'
print('name is {name}')
print('name is {}.'.format(name))
```
指定格式化的格式(specs)，可以这样：
```python
yes_votes = 42_572_654
no_votes = 43_132_495
percentage = yes_votes / (yes_votes + no_votes)
'{:-9} YES votes  {:2.2%}'.format(yes_votes, percentage)
```
fmtlib的设计，跟python的类似
```c++
fmt::print("elapsed time: {0:.2f} seconds". 1.23);
fmt::print(stderr, "Don't {}!", "panic");
```
之前的例子：
```c++
fmt::print("{:.2f}", 0.);
```

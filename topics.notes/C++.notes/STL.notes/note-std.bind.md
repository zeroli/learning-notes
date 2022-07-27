# std::bind/std::function的学习研究

```c++ {.numberLines}
    int a, b;
    std::function<void (int)> f = [&a](int) {
        std::cerr << (void*)&a << std::endl;
    };
    f(b);
    std::function<void (int)> f1;
    f1 = f;
    f1(b);
```

问题
===
上述输出一样的地址。
std::function的拷贝赋值操作是如何实现，特别是在lambda表达式绑定的是引用变量时。

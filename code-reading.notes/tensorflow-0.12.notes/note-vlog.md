# VLOG(x)宏实现的学习

```CPP
struct Voidifier {
  template <typename T>
  void operator&(const T&)const {}
};

#define VLOG(level) \
    (level > 0) ? (void)0 : Voidifier() & std::cout

const char* foo()
{
    std::cout << "calling foo function\n";
    return "calling foo";
}

int main()
{
    VLOG(1) << foo();

    return 0;
}
```
上述代码不会print任何输出，它的foo()函数不会被调用。
`VLOG(1) << foo();`会被展开为：
`(level > 0) ? (void)0 : Voidifier() & std::cout << foo();`
它的逻辑类似于这样：
```CPP
if (level > 0) {
    // nothing
} else {
    std::cout << foo();
}
```

- 如果level > 0，然后这条语句就是(void)0；啥都不干。
- 如果level <= 0，然后执行`Voidifier() & std::cout << foo();`。
`Voidifier`是一个类，`Voidifier()`就是构造一个对象，然后它后面的`&`，就是调用它的`operator &`操作符，这是个成员模板函数，可以接收任何参数，在这里就是`std::cout < foo();`的返回值，也就是std::cout对象。

`Voidifier() & std::cout << foo();`可以清楚表示为：
`Voidifier().operator & (std::cout << foo());`
- 构造Voidifier临时对象
- 调用std::cout <<操作符，进行输出，这是就会评估foo()函数
- 调用之前临时对象的operator &函数，接收std::cout，然会空(void)


**为啥要用这样的方式`Voidifier() & std::cout << foo();`，原因在于tenary operator要求第二和三操作数类型要相同，这里是第二个操作数类型是(void)0，第三个操作数返回类型也是void。**

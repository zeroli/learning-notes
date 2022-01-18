# 关于函数参数的传递

```c++ {.numberLines}
struct A {
    A() { std::cerr << "A ctor\n"; }
    ~A() { std::cerr << "A dtor\n"; }
    A(const A&) { std::cerr << "A copy-ctor\n"; }
    A& operator=(const A&) { std::cerr << "A copy-assignment\n"; return *this; }
    A(A&&) { std::cerr << "A move-ctor\n"; }
    A& operator=(A&&) { std::cerr << "A move-assignment\n"; return *this; }
    void print() const { std::cerr << "A::print\n"; }
};

template <typename T>
void foo1(T&& a) {
    T b = std::forward<T>(a);
    b.print();
}

void foo(A a) {
    foo1(a);
}

foo(A());
```
上述代码中，`foo(A())`，调用foo是传入一个临时构造的`A`对象，`foo(A a)`函数接收`A` by copy，然后再调用模板函数`foo1`，将`a`存在某处到`b`，之后进行调用。
输出如下:
```sh
A ctor
A::print
A dtor
```

但是如果我将`a`存在到另一个结构体里面，之后再调用它的什么函数，输入将会不一样了：
```c++
struct K {
    K(const A& a_): a(a_) { }
    A a;
};

template <typename T>
void foo2(T&& a) {
    K k(std::forward<T>(a));
    k.a.print();
}

void foo(A a) {
    foo1(a);
}

```
这时就会进行拷贝构造函数的调用

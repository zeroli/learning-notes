# C++ template tutorial

link: https://github.com/wuye9036/CppTemplateTutorial
这是一个关于C++ template的github博客，主要思想是说我们可以学习一门新语言的方式来学习c++ template。

template的基本语法
=====
模板类的声明：
```template <typename T> class ClassA;```
模板类的定义：
```C++
template <typename T>
class ClassA {
    T member;
};
```
- `template`是C++ keyword；
- template参数以`<>`括起来；
- template定义类似于函数定义：`void foo(int a)`
  - `typname`=> `int`, 类型匹配`T`;
  - `T` => `a`，参数名字;
  - `typename`甚至可以直接是`int`等；
- 接下来就是类的定义，在类的定义中，你还可以使用`T`这个特殊的模板参数类型；
譬如：
    `ClassA<int>`将类似于如下的C++伪代码定义：
    ```c++
    typedef class {
        int member；
    } ClassA<int>;
    ```
- 通过模板参数替换类型，可以获得很多形式相同的新类型，从而减少代码量；
  ==**泛型编程**== (generic programming)

模板的使用
====
```c++
template <typename T>
class vector {
    ...
    T* elements;
};
```
- 类型最重要的作用就是用它去产生一个变量；
- 变量定义的过程可以分两步来完成：
    1. `vector<int>`将`int`绑定到模板类`vector`上，获得一个普通的`vector<int>`类型；
    2. 通过`vector<int>`定义一个变量；
- 模板类是不能直接用来定义变量的，与普通类不同；
- 通过把类型绑定到模板类变成“普通类”的过程，称之为模板实例化(**template instantiate**);
    - `模板名 < 模板实参1 [, 模板实参2, ...] >`
    - 模板实参(类型)需要与模板形参正确匹配，否则不能正确的实例化；

模板类的成员函数定义
====
模板成员函数定义，一般以内联的方式实现：
示例:
```c++
template <typename T>
class vector
{
public:
    void clear()
    {
        // Function body
    }

private:
    T* elements;
};
```
也可以定义在类定义之外：
```c++
template <typename T>
class vector
{
public:
    void clear();  // 注意这里只有声明
private:
    T* elements;
};

template <typename T>  // 模板参数
void vector<T>::clear()  // 函数的实现放在这里
{
    // Function body
}
```

模板函数的语法
====
- 模板函数template function的声明和定义跟模板类的基本相同；
- 模板参数列表中的类型，可以出现在函数参数，返回值以及函数体内；
  ```c++
    template <typename T> void foo(T const& v);
    template <typename T> T foo();
    template <typename T, typename U> U foo(T const&);
    template <typename T> void foo()
    {
        T var;
        // ...
    }
    ```

与设计模式一样，模板在实际应用中，也会有一些固定的需求和解决方案；
- 泛型（基本用法）；
- 通过类型获取相应的信息（型别萃取）；
- 编译期的计算；
- 类型间的推到和变换（从一个类型变换成另外一个类型：`boost::function`）；

模板函数的使用
===
示例：
```c++
template <typename T> T Add(T a, T b)
{
    return a + b;
}
```
调用格式：`函数模板名  < 模板参数列表 > ( 函数实参列表 )`
```c++
int a = 5;
int b = 3;
int result = Add<int>(a, b);
```
这时相当于拥有了一个新函数：`int Add<int>(int a, int b) { return a + b; }`
compiler auto-deduction works; `Add(a, b);`
**但是编译器是无法根据返回值推断类型**，因为调用的时候，返回值被谁接受还不知道呢？
```c++
float data[1024];

template <typename T> T GetValue(int i)
{
    return static_cast<T>(data[i]);
}

float a = GetValue(0);	// 出错了！
int b = GetValue(1);	// 也出错了！
```
正确的调用如下：
```c++
float a = GetValue<float>(0);
int b = GetValue<int>(1);
```

`c_style_cast`示例，解释编译器推到部分模板参数：
`DstT dest = c_style_cast<DstT>(src);`
```c++
template <typename SrcT, typename DstT> DstT c_style_cast(SrcT v)
{
    return (DstT)(v);
}

int v = 0;
float i = c_style_cast<float>(v);
```
> error C2783: 'DstT _1_2_2::c_style_cast(SrcT)' : could not deduce template argument for 'DstT'
写上所有模板实参类型：`float i = c_style_cast<int, float>(v);`
也可以让编译器进行部分推导，但是编译器对模板参数的顺序是有限制的：
**先写需要指定的模板参数，再把能推导出来的模板参数放在后面**
```c++
template <typename DstT, typename SrcT> DstT c_style_cast(SrcT v)	// 模板参数 DstT 需要人肉指定，放前面。
{
    return (DstT)(v);
}

int v = 0;
float i = c_style_cast<float>(v);  // 形象地说，DstT会先把你指定的参数吃掉，剩下的就交给编译器从函数参数列表中推导啦。
```

整型模板参数
===
这里的整型比较宽泛：布尔型，不同位数、有无符号的整型，甚至包括指针；
```c++
template <typename T> class TemplateWithType;
template <int      V> class TemplateWithValue;
```
C++ template最初的想法：模板不就是为了提供一个类型安全，易于调试的宏吗？宏除了代码替换之后，还可以定义常数；
所以整型模板参数最基本的用途就是定义一个常数；
```c++
template <typename T, int Size> struct Array
{
    T data[Size];
};
Array<int, 16> arr;
```
相当于如下代码：
```c++
class IntArrayWithSize16
{
    int data[16]; // int 替换了 T, 16 替换了 Size
};

IntArrayWithSize16 arr;
```
整型参数是需要在编译期确定下来的；
几个复杂点的例子：
```c++
template <int i> class A
{
public:
    void foo(int)
    {
    }
};
template <uint8_t a, typename b, void* c> class B {};
template <bool, void (*a)()> class C {};
template <void (A<3>::*a)(int)> class D {};

template <int i> int Add(int a)	// 当然也能用于函数模板
{
    return a + i;
}

void foo()
{
    A<5> a;
    B<7, A<5>, nullptr>	b; // 模板参数可以是一个无符号八位整数，可以是模板生成的类；可以是一个指针。
    C<false, &foo> c;      // 模板参数可以是一个bool类型的常量，甚至可以是一个函数指针。
    D<&A<3>::foo> d;       // 丧心病狂啊！它还能是一个成员函数指针！
    int x = Add<3>(5);     // x == 8。因为整型模板参数无法从函数参数获得，所以只能是手工指定啦。
}

template <float a> class E {}; // ERROR: 别闹！早说过只能是整数类型的啦！
```
**整型模板参数除了这个最基本的用途之外，另外一个最重要的用途就是让类型可以像整数一样运算（book <<modern C++ design>>）**

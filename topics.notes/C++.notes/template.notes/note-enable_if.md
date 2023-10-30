# enable_if的学习

```c++
struct Foo {
    template < typename T,
               typename std::enable_if < !std::is_integral<T>::value, int >::type = 0 >
    void f(const T& value)
    {
        std::cout << "Not int" << std::endl;
    }

    template<typename T,
             typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    void f(const T& value)
    {
        std::cout << "Int" << std::endl;
    }
};

int main()
{
    Foo foo;
    foo.f(1);
    foo.f(1.1);

    // Output:
    // Int
    // Not int
}
```
- 这样也可以work
```
struct Foo {
    // SFINAE 在函数模板默认参数中
    template < typename T
        , typename std::enable_if < !std::is_integral<T>::value>::type* = nullptr
               >
    void f(const T& value)
    {
        std::cout << "Not int" << std::endl;
    }

    template<typename T
        , typename std::enable_if<std::is_integral<T>::value>::type* = nullptr
        >             
    void f(const T& value)
    {
        std::cout << "Int" << std::endl;
    }

    // SFINAE 在函数返回值类型中
    template < typename T>
    typename std::enable_if < !std::is_integral<T>::value>::type
     f2(const T& value)
    {
        std::cout << "Not int" << std::endl;
    }

    template < typename T>
    typename std::enable_if < std::is_integral<T>::value>::type
     f2(const T& value)  
    {
        std::cout << "Int" << std::endl;
    }

    // SFINAE 在函数默认参数中
    template < typename T>
    void f3(const T& value, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr)
    {
        std::cout << "Not int" << std::endl;
    }

    template < typename T>
    void f3(const T& value, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr)
    {
        std::cout << "Int" << std::endl;
    }
    
    /*  ==> f4不能定义为overload，compiler error
    template < typename T, typename = typename std::enable_if<!std::is_integral<T>::value>::type>
    void f4(const T& value)
    {
        std::cout << "Not int" << std::endl;
    }

    template < typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    void f4(const T& value)
    {
        std::cout << "Int" << std::endl;
    }
    */ 
};
```

有问题的enable_if用法：
```c++
template <class T>
class check
{
public:
   template< class U = T, class = typename std::enable_if<std::is_same<U, int>::value>::type >
   inline static U readVal()
   {
      return BuffCheck.getInt();
   }

   template< class U = T, class = typename std::enable_if<std::is_same<U, double>::value>::type >
   inline static U readVal()
   {
      return BuffCheck.getDouble();
   }
};
```
- `readVal`模板函数被定义了2次，模板参数都是U和另一个匿名的参数，不过后者有默认参数；
- 默认模板参数值(argument)并不是函数模板签名的一部分；
Default template arguments are not part of the signature of a template (so both definitions try to define the same template twice).
- Their parameter types are part of the signature.
  - 需要定义不同的模板参数，它根据已知的模板参数来进行判断是否可以导致整个函数被实例化存在；
- `SFINAE`在这里适用；
```c++
template <class T>
class check
{
public:
   // 如果U是int类型，这个函数被实例化，其它函数实例化失败
   template< class U = T,
             typename std::enable_if<std::is_same<U, int>::value, int>::type = 0>
   inline static U readVal()
   {
      return BuffCheck.getInt();
   }
   // 如果U是double类型，这个函数被实例化，其它函数实例化失败
   template< class U = T,
             typename std::enable_if<std::is_same<U, double>::value, int>::type = 0>
   inline static U readVal()
   {
      return BuffCheck.getDouble();
   }
};
```
另一个用法，enable_if通常被用在返回值类型判断中，从而决议某个函数被实例化出来；
```c++
template <class T>
class check
{
public:
   template< class U = T>
   static typename std::enable_if<std::is_same<U, int>::value, U>::type readVal()
   {
      return BuffCheck.getInt();
   }

   template< class U = T>
   static typename std::enable_if<std::is_same<U, double>::value, U>::type readVal()
   {
      return BuffCheck.getDouble();
   }
};
```
> 当enable_if的cond是false时，它是一个空类型，没有type，故而实例化出错，需要丢弃；

```c++
class foo;
class bar;

template<class T>
struct is_bar
{
    template<class Q = T>
    typename std::enable_if<std::is_same<Q, bar>::value, bool>::type check()
    {
        return true;
    }

    template<class Q = T>
    typename std::enable_if<!std::is_same<Q, bar>::value, bool>::type check()
    {
        return false;
    }
};

int main()
{
    is_bar<foo> foo_is_bar;
    is_bar<bar> bar_is_bar;
    if (!foo_is_bar.check() && bar_is_bar.check())
        std::cout << "It works!" << std::endl;

    return 0;
}
```

**c++17的代码更简洁:**
```c++
template <class T>
class check
{
public:
   inline static T readVal()
   {
        if constexpr (std::is_same_v<T, int>)
             return BuffCheck.getInt();
        else if constexpr (std::is_same_v<T, double>)
             return BuffCheck.getDouble();
   }
};
```

是否可以用类模板的偏特化来解决呢？
```c++
template <typename T> T CustomDiv(T lhs, T rhs) {
    T v;
    // Custom Div的实现
    return v;
}

template <
    typename T,
    typename Enabled = std::true_type
> struct SafeDivide {
    static T Do(T lhs, T rhs) {
        return CustomDiv(lhs, rhs);
    }
};

template <typename T> struct SafeDivide<T, typename std::is_floating_point<T>::type>{    // 偏特化A
    static T Do(T lhs, T rhs){
        return lhs/rhs;
    }
};

template <typename T> struct SafeDivide<T, typename std::is_integral<T>::type>{   // 偏特化B
    static T Do(T lhs, T rhs){
        return rhs == 0 ? 0 : lhs/rhs;
    }
};

void foo(){
    SafeDivide<float>::Do(1.0f, 2.0f);	// 调用偏特化A
    SafeDivide<int>::Do(1, 2);          // 调用偏特化B
    SafeDivide<std::complex<float>>::Do({1.f, 2.f}, {1.f, -2.f});
}
```
当你觉得需要写 enable_if 的时候，首先要考虑到以下可能性：
- 重载（对模板函数）
- 偏特化（对模板类而言）
- 虚函数

* Expression SFINAE (Substitution Failure Is Not An Error)
```c++
struct Counter {
    void increase() {
        // Implements
    }
};

template <typename T>
void inc_counter(T& intTypeCounter, std::decay_t<decltype(++intTypeCounter)>* = nullptr) {
    ++intTypeCounter;
}

template <typename T>
void inc_counter(T& counterObj, std::decay_t<decltype(counterObj.increase())>* = nullptr) {
    counterObj.increase();
}

void doSomething() {
    Counter cntObj;
    uint32_t cntUI32;

    // blah blah blah
    inc_counter(cntObj);
    inc_counter(cntUI32);
}
```
Expression SFINAE是C++才引入进来的，Type SFINAE在C++11之前就存在；

```c++
struct ICounter {};
struct Counter: public ICounter {
    void increase() {
        // impl
    }
};

template <typename T> void inc_counter(
    T& counterObj,
    typename std::enable_if<
        std::is_base_of<ICounter, T>::value
    >::type* = nullptr ){
    counterObj.increase();
}

template <typename T> void inc_counter(
    T& counterInt,
    typename std::enable_if<
        std::is_integral<T>::value
    >::type* = nullptr ){
    ++counterInt;
}

void doSomething() {
    Counter cntObj;
    uint32_t cntUI32;

    // blah blah blah
    inc_counter(cntObj); // OK!
    inc_counter(cntUI32); // OK!
}
```

# copy-and-swap pattern to rule-of-five for class definition

```c++ {.numberLines}
#include <utility>

class resource {
  int x = 0;
};

class foo
{
  public:
    foo()
      : p{new resource{}}
    { }

    foo(const foo& other)
      : p{new resource{*(other.p)}}
    { }

    foo(foo&& other)
      : p{other.p}
    {
      other.p = nullptr;
    }

    foo& operator=(foo other)
    {
      swap(*this, other);

      return *this;
    }

    ~foo()
    {
      delete p;
    }

    friend void swap(foo& first, foo& second)
    {
      using std::swap;

      swap(first.p, second.p);
    }

  private:
    resource* p;
};
```

Requires c++11 or newer.
INTENT
Implement the assignment operator with strong exception safety.

DESCRIPTION
The copy-and-swap idiom identifies that we can implement a classes copy/move assignment operators in terms of its copy/move constructor and achieve strong exception safety.

**The class foo, on lines 7–45, has an implementation similar to the rule of five, yet its copy and move assignment operators have been replaced with a single assignment operator on lines 24–29. This assignment operator takes its argument by value, making use of the existing copy and move constructor implementations.**
注意这里，拷贝赋值和移动赋值被一个函数替换：
通常我们些拷贝赋值，是这样的:
```c++
foo& operator=(const foo& other)
{
    foo(other).swap(*this);
    return *this;
}
foo& operator=(foo& other)
{
    foo(std::move(other)).swap(*this);
    return *this;
}
```
上述代码`other`都会被用来初始化一个临时的foo，然后再于`*this`进行交换；
这样的话，前面案例中的参数定义为`foo other`，会能够统一对通常的2个函数定义。

To implement the assignment operator, we simply need to swap the contents of *this and the argument, other. When other goes out of scope at the end of the function, it will destroy any resources that were originally associated with the current object.

To achieve this, we define a swap function for our class on lines 36–41, which itself calls swap on the class’s members (line 40). We use a using-declaration on line 38 to allow swap to be found via argument-dependent lookup before using std::swap — this is not strictly necessary in our case, because we are only swapping a pointer, but is good practice in general. Our assignment operator then simply swaps *this with other on line 26.

The copy-and-swap idiom has inherent strong exception safety because all allocations (if any) occur when copying into the other argument, before any changes have been made to *this. It is generally, however, less optimized than a more custom implementation of the assignment operators.

Note: We can typically avoid manual memory management and having to write the copy/move constructors, assignment operators, and destructor entirely by using the rule of zero

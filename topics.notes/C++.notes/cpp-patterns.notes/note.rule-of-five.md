# rule of five for one class

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

    foo& operator=(const foo& other)
    {
      if (&other != this) {
        delete p;
        p = nullptr;
        p = new resource{*(other.p)};
      }
      return *this;
    }

    foo& operator=(foo&& other)
    {
      if (&other != this) {
        delete p;
        p = other.p;
        other.p = nullptr;
      }
      return *this;
    }

    ~foo()
    {
      delete p;
    }

  private:
    resource* p;
};
```

Requires c++11 or newer.
INTENT
Safely and efficiently implement RAII to encapsulate the management of dynamically allocated resources.

DESCRIPTION
The rule of five is a modern expansion of the rule of three. Firstly, the rule of three specifies that if a class implements any of the following functions, it should implement all of them:

- copy constructor
- copy assignment operator
- destructor

These functions are usually required only when a class is manually managing a dynamically allocated resource, and so all of them must be implemented to manage the resource safely.

In addition, the rule of five identifies that it usually appropriate to also provide the following functions to allow for optimized copies from temporary objects:

- move constructor
- move assignment operator

The class foo, on lines 7–53, dynamically allocates a resource object in its constructor. The implementations of foo’s copy constructor (lines 14–16), copy assignment operator (lines 24–33), and destructor (lines 46–49) ensure that the lifetime of this resource is safely managed by foo object that contains it, even in the event of an exception.

We have also implemented a move constructor (lines 18–22) and move assignment operator (lines 35–44) that provide optimized copies from temporary objects. Rather than copy the resource, they take the resource from the original foo and set its internal pointer to nullptr, effectively stealing the resource.

Notice that the assignment operators (lines 24–44) check for self-assignment to ensure safe management of the resource.

Note: The copy and move assignment operators in the example code provide only basic exception safety. They may alternatively be implemented with the copy-and-swap idiom, which provides strong exception safety at an optimisation cost.

Note: We can typically avoid manual memory management and having to write the copy constructor, assignment operator, and destructor entirely by using the rule of zero

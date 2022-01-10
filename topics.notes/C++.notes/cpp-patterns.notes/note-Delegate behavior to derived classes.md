# Delegate behavior to derived classes

**Curiously Recurring Template Pattern (CRTP)**

```c++ {.numberLines}
template<typename derived>
class base
{
  public:
    void do_something()
    {
      // ...
      static_cast<derived*>(this)->do_something_impl();
      // ...
    }

  private:
    void do_something_impl()
    {
      // Default implementation
    }
};
class foo : public base<foo>
{
  public:
    void do_something_impl()
    {
      // Derived implementation
    }
};

class bar : public base<bar>
{ };

template<typename derived>
void use(base<derived>& b)
{
  b.do_something();
}
```

Requires c++98 or newer.
INTENT
Delegate behavior to derived classes without incurring the cost of run-time polymorphism.

DESCRIPTION
With the Curiously Recurring Template Pattern (CRTP), which provides a form of static polymorphism, we can delegate behavior from a base class to its derived classes. This approach avoids the costs associated with using virtual functions for run-time polymorphism, typically implemented with a virtual function table (a dynamic dispatch mechanism).

Classes foo and bar, on lines 19–29, demonstrate the CRTP idiom by inheriting from the base class template (lines 1–17) and providing themselves as the template argument. For example, foo inherits from base<foo> on line 19. This allows base to know which class it is being inherited by at compile-time.

The base class provides a public member function, do_something (lines 5–10), which depends on do_something_impl, an internal function that may optionally be overriden by derived classes. In this way, base is able to delegate behavior to derived classes. A default implementation for this function is given on lines 13–16, while the class foo provides its own implementation on lines 22–25. To ensure that the correct implementation is used, the do_something function casts this to a pointer to the derived type on line 8 and calls do_something_impl on it.

The use function template on lines 31–35 takes a reference to any instantiation of base and calls do_something on it. As the derived type is known at compile-time, the correct implementation function is called without the need for dynamic dispatch. If a base<foo> is provided, for example, foo’s implementation (lines 22–25) will be invoked. For a base<bar>, on the other hand, the default implementation defined by base will be used (lines 13–16).

# visitor模式

```c++ {.numberLines}
class element_concrete_1;
class element_concrete_2;
class visitor
{
  public:
    virtual void visit(element_concrete_1& el) = 0;
    virtual void visit(element_concrete_2& el) = 0;
};
class visitor_concrete : public visitor
{
  public:
    virtual void visit(element_concrete_1& el) override
    {
      // Do something with el
    };
    virtual void visit(element_concrete_2& el) override
    {
      // Do something with el
    };
};
class element
{
  public:
    virtual void accept(visitor& v) = 0;
};
class element_concrete_1 : public element
{
  public:
    virtual void accept(visitor& v) override
    {
      v.visit(*this);
    }
};
class element_concrete_2 : public element
{
  public:
    virtual void accept(visitor& v) override
    {
      v.visit(*this);
    }
};
```
Requires c++98 or newer.
INTENT
Separate generic algorithms from the elements or structure on which they operate.

DESCRIPTION
The visitor pattern allows generic algorithms to be implemented without modifying the objects on which they operate and supports different actions for each type of object without the need for dynamic casting.

The element class, defined on lines 26–30, defines an interface that allows an object to accept a visitor. Each class derived from element, such as those on lines 32–48, implements the accept function such that it calls the appropriate visit function on the visitor.

Visitors, which implement the visitor interface (lines 4–9), represent algorithms to be applied to elements. The visitor interface defines visit overloads for each type of element. An example visitor is defined on lines 11–23.

When a visitor visits an element, two function calls are made (one to accept and one to visit) and the final visit function that is called depends on the type of both the element and the visitor. This process is known as double dispatch.

The visitor pattern is particularly useful when the elements are part of a larger structure, in which case the accept function can call itself recursively down the structure.

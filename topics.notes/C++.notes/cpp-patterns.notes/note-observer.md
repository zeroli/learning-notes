# observer

```c++ {.numberLines}
#include <vector>
#include <functional>
class observer
{
  public:
    virtual void notify() = 0;
};
class observer_concrete : public observer
{
  public:
    virtual void notify() override
    { }
};
class subject
{
  public:
    void register_observer(observer& o)
    {
      observers.push_back(o);
    }
    void notify_observers()
    {
      for (observer& o : observers) {
        o.notify();
      }
    }
  private:
    std::vector<std::reference_wrapper<observer>> observers;
};
```

Requires c++11 or newer.
INTENT
Notify generic observer objects when an event occurs.

DESCRIPTION
The observer pattern allows generic observer objects to be registered with a subject object and receive notifications when certain events occur.

The subject class, defined on lines 17–34, contains a std::vector of references to observers line 33. Observers (also known as listeners), in this case, are objects that implement the observer interface (lines 4–8). The register_observer function (lines 20–23) adds observers to this std::vector, which are later to be notified by the notify_observers function (lines 25–30).

We use std::reference_wrapper for the elements of the std::vector (line 33), because the standard containers require the element type to be assignable, which normal reference types are not.

==>注意这里的`std::reference_wrapper<observer>`的使用

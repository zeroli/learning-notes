# Avoid std::bind
This Tip summarizes reasons why you should stay away from std::bind() when writing code.

Using std::bind() correctly is hard. Let’s look at a few examples. Does this code look good to you?
```c++
void DoStuffAsync(std::function<void(absl::Status)> cb);

class MyClass {
  void Start() {
    DoStuffAsync(std::bind(&MyClass::OnDone, this));
  }
  void OnDone(absl::Status status);
};
```
Many seasoned C++ engineers have written code like this only to find that it doesn’t compile. It works with std::function<void()> but breaks when you add an extra parameter to MyClass::OnDone. What gives?

std::bind() doesn’t just bind the first N arguments like many C++ engineers expect (this is called partial function application). You must instead specify every argument, so the correct incantation of std::bind() is this:

std::bind(&MyClass::OnDone, this, std::placeholders::_1)
Ugh, that’s ugly. Is there a better way? Why yes, use absl::bind_front() instead.

absl::bind_front(&MyClass::OnDone, this)
Remember partial function application – the thing that std::bind() does not do? Well, absl::bind_front() does exactly that: it binds the first N arguments and perfect-forwards the rest: absl::bind_front(F, a, b)(x, y) evaluates to F(a, b, x, y).

Ahhh, sanity is restored. Want to see something truly terrifying now? What does this code do?
```c++
void DoStuffAsync(std::function<void(absl::Status)> cb);

class MyClass {
  void Start() {
    DoStuffAsync(std::bind(&MyClass::OnDone, this));
  }
  void OnDone();  // No absl::Status here.
};
```
OnDone() takes no parameters, the DoStuffAsync() callback should take a absl::Status. You might expect a compile error here but this code actually compiles without warnings, because std::bind over-aggressively bridges the gap. Potential errors from DoStuffAsync() get silently ignored.

Code like this can cause serious damage. Thinking that some IO has succeeded when it actually hasn’t can be devastating. Perhaps the author of MyClass didn’t realize that DoStuffAsync() may produce an error that needs to be handled. Or maybe DoStuffAsync() used to accept std::function<void()> and then its author decided to introduce the error mode and updated all callers that stopped compiling. Either way the bug made its way into production code.

std::bind() disables one of the basic compile time checks that we all rely on. The compiler usually tells you if the caller is passing more arguments than you expect, but not for std::bind(). Terrifying enough?

Another example. What do you think of this one?
```c++
void Process(std::unique_ptr<Request> req);

void ProcessAsync(std::unique_ptr<Request> req) {
  thread::DefaultQueue()->Add(
      ToCallback(std::bind(&MyClass::Process, this, std::move(req))));
}
```
Good old passing std::unique_ptr across async boundaries. Needless to say, std::bind() isn’t a solution – the code doesn’t compile, because std::bind() doesn’t move the bound move-only argument to the target function. Simply replacing std::bind() with absl::bind_front() fixes it.

The next example regularly trips even C++ experts. See if you can find the problem.
```c++
// F must be callable without arguments.
template <class F>
void DoStuffAsync(F cb) {
  auto DoStuffAndNotify = [](F cb) {
    DoStuff();
    cb();
  };
  thread::DefaultQueue()->Schedule(std::bind(DoStuffAndNotify, cb));
}

class MyClass {
  void Start() {
    DoStuffAsync(std::bind(&MyClass::OnDone, this));
  }
  void OnDone();
};
```
This doesn’t compile because passing the result of std::bind() to another std::bind() is a special case. Normally, std::bind(F, arg)() evaluates to F(arg), except when arg is the result of another std::bind() call, in which case it evaluates to F(arg()). If arg is converted to std::function<void()>, the magic behavior is lost.

Applying std::bind() to a type you don’t control is always a bug. DoStuffAsync() shouldn’t apply std::bind() to the template argument. Either absl::bind_front() or lambda would work fine.

The author of DoStuffAsync() might even have entirely green tests because they always pass a lambda or std::function as the argument but never the result of std::bind(). The author of MyClass will be puzzled when hitting on this bug.

Is this special case of std::bind() ever useful? Not really. It only gets in the way. If you are composing functions by writing nested std::bind() calls, you really should write a lambda or a named function instead.

Hopefully you are convinced that it’s easy to go wrong with std::bind(). Runtime and compile time traps are easy to fall into both for newcomers and C++ experts. Now I’ll try to show that even when std::bind() is used correctly, there is usually another option that is more readable.

Calls to std::bind() without placeholders are better off as lambdas.
```c++
std::bind(&MyClass::OnDone, this)
```
vs
```c++
[this]() { OnDone(); }
```
Calls to std::bind() that perform partial application are better off as absl::bind_front(). The more placeholders you have, the more obvious it gets.

std::bind(&MyClass::OnDone, this, std::placeholders::_1)
vs

absl::bind_front(&MyClass::OnDone, this)
(Whether to use absl::bind_front() or a lambda when performing partial function application is a judgement call; use your discretion.)

This covers 99% of all calls std::bind(). The remaining calls do something fancy:

Ignore some of the arguments: std::bind(F, _2).
Use the same argument more than once: std::bind(F, _1, _1).
Bind arguments at the end: std::bind(F, _1, 42).
Change argument order: std::bind(F, _2, _1).
Use function composition: std::bind(F, std::bind(G)).
These advanced uses may have their place. Before resorting to them, consider all known issues with std::bind() and ask yourself whether the potential savings of a few characters or lines of code are worth it.

Conclusion
Avoid std::bind. Use a lambda or absl::bind_front instead.

Further Reading
'’Effective Modern C++’’, Item 34: Prefer lambdas to std::bind.

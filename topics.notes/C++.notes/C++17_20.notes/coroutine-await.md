* C++20 introduced `co_await` operator (similar to new operator)
so that it could be overridden/re-implemented
```c++
struct dummy { // Awaitable
    std::suspend_always operator co_await(){ return {}; }
};

HelloWorldCoro print_hello_world() {
    std::cout << "Hello ";
    co_await dummy{}; 
    std::cout << "World!" << std::endl;
}
```

Similar to `new` operator, since if we re-imlement `new` operator for one class
and call `new MyClass(...);` then `new` operator will use `new` method defined in MyClass
 to allocate memory, instead of global `new` operator
Here, we use `co_await dummy{};` to call into `dummy` class's customized `co_await` operator

# Using C++20 corountine with handy code Generator to generate a sequence of numbers
* use `g++ -std=c++20 -fno-exceptions` to compile
* `-fno-exceptions` is needed to remove compilation error:
    >> no member named 'unhandled_exception' 

```c++
#include <coroutine>
#include <iostream>
#include <iterator>

struct Generator {
    struct promise_type {
        int d_val;

        Generator get_return_object() { return Generator{this}; }
        // should be suspend_never (don't suspend) so that we will not 
        // stop after make coroutine, and yield first value which will call
        // `yield_value(start)`
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        std::suspend_always yield_value(int val) noexcept {
            d_val = val;
            return {};
        }
    };

    struct Iterator {
        bool operator !=(const Iterator&) { return !d_hdl->done(); }
        // when iterator advances, we'd like to ask coroutine to generate
        // new value, so `resume` called to resume coroutine
        Iterator& operator++() {
            d_hdl->resume();
            return *this;
        }
        int operator *() {            
            return d_hdl->promise().d_val;
        }
        std::coroutine_handle<promise_type>* d_hdl;
    };

    Iterator begin() { 
        return Iterator{&d_hdl};
    }
    Iterator end() {
        return Iterator{nullptr};
    }

    Generator(promise_type* promise)
        : d_hdl(std::coroutine_handle<promise_type>::from_promise(*promise))
    { }
    ~Generator() {
        d_hdl.destroy();
    }
    std::coroutine_handle<promise_type> d_hdl;
};

Generator range(int start, int end) {
    while (start != end) {
        co_yield start++;
    }
}

int main()
{
    for (auto&& v : range(0, 10)) {
        std::cout << v << ",";
    }
    std::cout << "\n";
}
```

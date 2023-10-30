#include <utility>
#include <type_traits>
#include <iostream>
#include <functional>
#include <string>

struct Status {
    int error;
    std::string msg;
};

void foo() { }
int foo1() { return 0; }
int foo2(int x) { return x; }
struct A {
    A() {
        std::cout << "A ctor\n";
    }
    ~A() {
        std::cout << "A dtor\n";
    }
    A(A&&) {
        std::cout << "A move ctor\n";
    }
    A& operator =(A&&) {
        std::cout << "A move assignment\n";
        return *this;
    }
    A(const A&) {
        std::cout << "A copy ctor\n";
    };
    A& operator=(const A&) {
        std::cout << "A copy assignment\n";
        return *this;
    }
};

A foo3() { return A{}; }

struct Callable {
	int operator ()(int x) const {
      return x * 2;
    }
  	void operator()() const {
      return;
    }
};

template <typename T>
struct StatusOr {
    Status status;
    T value;
    explicit StatusOr(const T& val)
        : value(val)
    { }

    operator T() const {
        return value;
    }
};

template <>
struct StatusOr<void> {
    explicit StatusOr(...) { }   
};

// this is also working together with below variadic template version
/*
template <typename R, typename... Args>
R submit(R (*f)(Args...), Args&&... args)
{
    return f(std::forward<Args>(args)...);
}
*/

template <typename F, typename... Args>
typename std::enable_if<
    std::is_same<void, typename std::result_of<F(Args&&...)>::type>::value
    >::type
submit(F&& f, Args&&... args)
{
    using Ret_t = typename std::result_of<F(Args&&...)>::type;
    f(std::forward<Args>(args)...);
}

template <typename F, typename... Args>
typename std::enable_if<
    !std::is_same<void, typename std::result_of<F(Args&&...)>::type>::value,
    StatusOr<typename std::result_of<F(Args&&...)>::type>
    >::type
submit(F&& f, Args&&... args)
{
    using Ret_t = typename std::result_of<F(Args&&...)>::type;
    return StatusOr<Ret_t>(f(std::forward<Args>(args)...));
}



int main()
{
  	submit(Callable{}, 10);
    submit(Callable{});
    submit(&foo);
    submit(foo);
    std::cout << submit(foo1) << "\n";
    std::cout << submit(foo2, 10) << "\n";
    auto a = submit(foo3);
    auto b = a;

    submit([] {
        std::cout << "this is a lambda expression\n";
    });
    std::cout << submit([](int x, int y) {
        std::cout << "this is a lambda expression: " << x << "," << y << "\n";
      	return x + y;
    }, 30, 10) << "\n";

    int x = 30, y = 10;
    submit([x, y] {
        std::cout << "this is another lambda with capture: " << x << ", " << y << "\n";
        return x + y;
    });

    submit(std::bind(foo));
  
  {
  	std::function<void ()> f = foo;
   	submit(f); 
  }
   {
  	std::function<int ()> f = foo1;
   	submit(f); 
  }
}
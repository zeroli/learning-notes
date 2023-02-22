# 折叠表达式求值 (C++17)
====

* 代码练习

```c++
template <typename firstArg, typename... Args>
void print(firstArg firstarg, Args... args)
{
    std::cout << firstarg;
    auto spaceBefore = [](const auto& arg) {
        std::cout << " "  << arg;        
    };
    (... , spaceBefore(args));
    std::cout << "\n";
}

template <typename... Bases>
struct MultiBases : Bases...
{
    void print() {
        (..., Bases::print());
    }
};

struct A { void print() { std::cout << "A\n"; }};
struct B { void print() { std::cout << "B\n"; }};
struct C { void print() { std::cout << "C\n"; }};

template <typename T, typename... Tn>
struct IsHamogenous {
    static constexpr bool value = (... && std::is_same_v<T, Tn>);
};

template <typename T, typename... Tn>
bool IsHamogenous_f(T, Tn...)
{
    return (... && std::is_same_v<T, Tn>);
}

int main()
{
    print(1, 2, 3, "hello", std::string("world"));
    MultiBases<A, B, C>().print();
    std::cout << IsHamogenous<int, int, int>::value << "\n";
    std::cout << IsHamogenous_f(1, 2, "hello") << "\n";
}
```

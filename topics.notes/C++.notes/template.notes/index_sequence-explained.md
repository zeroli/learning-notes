```c++
template <typename T, T... Ints>
struct integer_sequence
{ };

template <std::size_t... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>;
// 实例化: index_sequce<0, 1, 2, 3> for example

template <typename T, std::size_t N, T... Is>
struct make_integer_sequence :
    make_integer_sequence<T, N-1, N-1, Is...>
{ };

template <typename T, T... Is>
struct make_integer_sequence<T, 0, Is...> :
    integer_sequence<T, Is...>
{ };
/*
每个类make_integer_sequence偏特化时绑定一个整型index，把它收集到Is中
make_integer_sequence with N => 收集n-1
父类make_integer_sequence with N-1 => 收集n-2
父类make_integer_sequence with N-2 => 收集n-3
...
顶层父类make_integer_sequence with 0 => 收集0
最后产生integer_sequence<size_t, 0, 1, 2, 3, 4, ..., n-1>的实例化的类
在编译期，由N产生这样的序列化的整型序列
*/
template <size_t N>
using make_integer_sequence = make_integer_sequence<size_t, N>;

template <typename... Ts>
using index_sequence_for = make_integer_sequence<sizeof...(Ts)>;

```
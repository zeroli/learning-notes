```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>

using namespace std;

template <typename... Ts>
struct TypeList { };

// length of typelist
namespace detail {
template <typename T>
struct length_list;

template <typename... Ts>
struct length_list<TypeList<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)>
{
};

}  // namespace detail

template <typename... Ts>
using length_t = typename detail::length_list<TypeList<Ts...>>::type;

template <typename... Ts>
constexpr size_t length_v = length_t<Ts...>::value;

struct empty_type { };
// at_t: one type at index of typelist
namespace detail {
template <size_t I, size_t N, typename T>
struct at_list;

template <size_t I, size_t N, typename U, typename... Ts>
struct at_list<I, N, TypeList<U, Ts...>> {
    using type =
        std::conditional_t<
            I == N,
            U,
            typename at_list<I, N+1, TypeList<Ts...>>::type>;
};

// end of recursion
template <size_t I, size_t N>
struct at_list<I, N, TypeList<>> {
    using type = empty_type;
};
}  // namespace detail

template <size_t I, typename... Ts>
using at_t = typename detail::at_list<I, 0, TypeList<Ts...>>::type;

// front_t: front type of typelist
namespace detail {
template <typename T>
struct front_list;

template <typename U, typename... Ts>
struct front_list<TypeList<U, Ts...>> {
    using type = U;
};

template <>
struct front_list<TypeList<>> {
    using type = empty_type;
};
}  // namespace detail

template <typename... Ts>
using front_t = typename detail::front_list<TypeList<Ts...>>::type;

// back_t: back type of TypeList
namespace detail {
template <int N, typename T>
struct back_list;

template <int N, typename U, typename... Ts>
struct back_list<N, TypeList<U, Ts...>> {
    using type = typename back_list<N-1, TypeList<Ts...>>::type;
};

template <typename U>
struct back_list<0, TypeList<U>> {
    using type = U;
};

template <typename... Ts>
struct back_list<-1, TypeList<Ts...>> {
    using type = empty_type;
};

}  // namespace detail

template <typename... Ts>
using back_t = typename detail::back_list<static_cast<int>(sizeof...(Ts)) - 1, TypeList<Ts...>>::type;

// push_back_t/push_front_t: add one more type to TypeList
namespace detail {
template <typename TL, typename T>
struct push_back_list;

template <template <typename... Ts> class TL, typename T, typename... Ts>
struct push_back_list<TL<Ts...>, T> {
    using type = TL<Ts..., T>;  
};

template <typename TL, typename T>
struct push_front_list;

template <template <typename... Ts> class TL, typename T, typename... Ts>
struct push_front_list<TL<Ts...>, T> {
    using type = TL<T, Ts...>;  
};
}  // namespace detail

template <typename TL, typename T>
using push_back_t = typename detail::push_back_list<TL, T>::type;

template <typename TL, typename T>
using push_front_t = typename detail::push_front_list<TL, T>::type;

// pop_back/pop_front: remove one type from TypeList
namespace detail {
template <int N, typename TL, typename TL1>
struct pop_back_list;

template <int N, template <typename...> class TL, typename T, 
    template <typename...> class TL1, typename... Ts, typename... Ts1>
struct pop_back_list<N, TL<T, Ts...>, TL1<Ts1...>> {
    using type = std::conditional_t<
        N == 0,
        TL1<Ts1...>,
        typename pop_back_list<
            N-1,
            TL<Ts...>,
            TL1<Ts1..., T>  // move T to TL1 from TL
            >::type
        >;
};

template <template <typename...> class TL, template <typename...> class TL1,
        typename... Ts1>
struct pop_back_list<-1, TL<>, TL1<Ts1...>> {
    using type = TL1<Ts1...>;
};


template <typename TL>
struct pop_front_list;

template <template <typename...> class TL, typename T, typename... Ts>
struct pop_front_list<TL<T, Ts...>> {
    using type = TL<Ts...>;
};

template <template <typename...> class TL>
struct pop_front_list<TL<>> {
    using type = TL<>;
};

}  // namespace detail

template <typename TL>
using pop_back_t = typename detail::pop_back_list<static_cast<int>(detail::length_list<TL>::value)-1, TL, TypeList<>>::type;

template <typename TL>
using pop_front_t = typename detail::pop_front_list<TL>::type;

int main()
{
    static_assert(length_v<int> == 1);
    static_assert(length_v<int, double> == 2);
    static_assert(std::is_same_v<int, at_t<0, int, double>>);
    static_assert(std::is_same_v<double, at_t<1, int, double>>);
    static_assert(std::is_same_v<empty_type, at_t<3, int, double>>);

    static_assert(std::is_same_v<empty_type, front_t<>>);
    static_assert(std::is_same_v<int, front_t<int, double>>);
    
    static_assert(std::is_same_v<empty_type, back_t<>>);
    static_assert(std::is_same_v<double, back_t<int, double>>);
    static_assert(std::is_same_v<float, back_t<int, double, float>>);

    static_assert(std::is_same_v<TypeList<int, double>, push_back_t<TypeList<int>, double>>);
    static_assert(std::is_same_v<TypeList<int>, push_back_t<TypeList<>, int>>);

    static_assert(std::is_same_v<TypeList<double, int>, push_front_t<TypeList<int>, double>>);
    static_assert(std::is_same_v<TypeList<int>, push_front_t<TypeList<>, int>>);
    
    static_assert(std::is_same_v<TypeList<int>, pop_back_t<TypeList<int, double>>>);
    static_assert(std::is_same_v<TypeList<>, pop_back_t<TypeList<int>>>);

    static_assert(std::is_same_v<TypeList<double>, pop_front_t<TypeList<int, double>>>);
    static_assert(std::is_same_v<TypeList<>, pop_front_t<TypeList<int>>>);

    return 0;
}
```

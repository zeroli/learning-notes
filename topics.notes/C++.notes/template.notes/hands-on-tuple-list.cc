#include <iostream>
#include <type_traits>
#include <string>
#include <memory>

template <typename ...>
struct Tuple;

// basic case for recursive
template <>
struct Tuple<> {};

template <typename Head, typename... Tails>
struct Tuple<Head, Tails...> {
    Head head_;
    Tuple<Tails...> tails_;

    Tuple() = default;

    /// add enable_if SFINAE to not resolve to this ctor
    /// Tuple<...> t1(t);  // t is one Tuple
    /// which is a copy ctor
    template <typename H, typename... Ts,
        std::enable_if_t<sizeof...(Ts) == sizeof...(Tails)>* = nullptr>
    Tuple(H&& h, Ts&&... ts)
        : head_(std::forward<H>(h)), tails_(std::forward<Ts>(ts)...)
    {
    }
    
    template <typename UH, typename... UTs,
        std::enable_if_t<sizeof...(UTs) == sizeof...(Tails)>* = nullptr>
    Tuple(const Tuple<UH, UTs...>& other)
        : head_(other.getHead()), tails_(other.getTails())
    {
    }   

    Head& getHead() { return head_; }
    const Head& getHead() const { return head_; }
    Tuple<Tails...>& getTails() { return tails_; }
    const Tuple<Tails...>& getTails() const { return tails_; }
};

template <size_t Idx, typename TupleT>
struct getHelper;

template <typename H, typename... Ts>
struct getHelper<0, Tuple<H, Ts...>>
{
    static auto& apply(const Tuple<H, Ts...>& t) {
        return t.getHead();
    }
};

template <size_t Idx, typename H, typename... Ts>
struct getHelper<Idx, Tuple<H, Ts...>>
{
    static auto& apply(const Tuple<H, Ts...>& t) {
        return getHelper<Idx-1, Tuple<Ts...>>::apply(t.getTails());
    }
};

template <size_t Idx, typename TupleT>
auto& get(const TupleT& t)
{
    return getHelper<Idx, TupleT>::apply(t);
}

namespace detail {
template <size_t N, typename... Ts>
struct printTuple;

template <>
struct printTuple<0> {
    static void apply(std::ostream& os, const Tuple<>& t) { }
};

template <size_t N, typename H, typename... Ts>
struct printTuple<N, H, Ts...> {
    static void apply(std::ostream& os, const Tuple<H, Ts...>& ts) {
        os << ts.getHead();
        if (N > 1) os << ", ";
        printTuple<N-1, Ts...>::apply(os, ts.getTails());
    }
};
}  // namespace detail

template <typename... Ts>
std::ostream& operator <<(std::ostream& os, const Tuple<Ts...>& t)
{
    os << "(";
    detail::printTuple<sizeof...(Ts), Ts...>::apply(os, t);
    os << ")";
    return os;
}

template <typename Tuple>
struct Front;
template <typename H, typename... Ts>
struct Front<Tuple<H, Ts...>> {
    using type = H;
};
template <typename Tuple>
using Front_t = typename Front<Tuple>::type;

template <typename Tuple>
struct PopFront;
template <typename H, typename... Ts>
struct PopFront<Tuple<H, Ts...>> {
    using type = Tuple<Ts...>;
};
template <typename Tuple>
using PopFront_t = typename PopFront<Tuple>::type;

template <typename Tuple, typename NewT>
struct PushFront;
template <typename... Ts, typename NewT>
struct PushFront<Tuple<Ts...>, NewT> {
    using type = Tuple<NewT, Ts...>;
};
template <typename Tuple, typename NewT>
using PushFront_t = typename PushFront<Tuple, NewT>::type;

template <typename Tuple, typename NewT>
struct PushBack;
template <typename... Ts, typename NewT>
struct PushBack<Tuple<Ts...>, NewT> {
    using type = Tuple<Ts..., NewT>;
};
template <typename Tuple, typename NewT>
using PushBack_t = typename PushBack<Tuple, NewT>::type;

template <typename Tuple>
struct Reverse;
template <>
struct Reverse<Tuple<>> { using type = Tuple<>; };

template <typename H, typename... Ts>
struct Reverse<Tuple<H, Ts...>> {
    /// A B C => A (B C) => (C B) A
    using type = typename PushBack<typename Reverse<Tuple<Ts...>>::type, H>::type;
};

template <typename Tuple>
using Reverse_t = typename Reverse<Tuple>::type;

template <typename Tuple>
struct PopBack;
template <typename... Ts>
struct PopBack<Tuple<Ts...>> {
    using type = Reverse_t<PopFront_t<Reverse_t<Tuple<Ts...>>>>;
};
template <typename Tuple>
using PopBack_t = typename PopBack<Tuple>::type;

template <typename... Ts>
auto makeTuple(Ts&&... ts)
{
    return Tuple<std::decay_t<Ts>...>{std::forward<Ts>(ts)...};
}

// compile-time `t` literal is encoded in CTValue parametrized class
template <typename T, T t>
struct CTValue {
    static constexpr T value = t;
};  

// compile-time `ts` literals are encoded in ValueList parametrized class
template <typename T, T... ts>
struct ValueList {
    using type = T;
};

template <typename ValueList>
struct FrontT;
template <typename T, T h, T... ts>
struct FrontT<ValueList<T, h, ts...>> {
    using type = CTValue<T, h>;
    static constexpr T value = h;
};

template <typename ValueList>
struct PopFrontT;
template <typename T, T h, T... ts>
struct PopFrontT<ValueList<T, h, ts...>> {
    using type = ValueList<T, ts...>;
};

template <typename ValueList, typename CTValue>
struct PushFrontT;
template <typename T, T... ts, T NewV>
struct PushFrontT<ValueList<T, ts...>, CTValue<T, NewV>> {
    using type = ValueList<T, NewV, ts...>;
};

template <typename ValueList, typename CTValue>
struct PushBackT;
template <typename T, T... ts, T NewV>
struct PushBackT<ValueList<T, ts...>, CTValue<T, NewV>> {
    using type = ValueList<T, ts..., NewV>;
};

template <size_t N, typename Result = ValueList<unsigned>>
struct MakeIndexListT
    : MakeIndexListT<N-1, typename PushFrontT<Result, CTValue<unsigned, N-1>>::type>
{};

template <typename Result>
struct MakeIndexListT<0, Result>
{
    using type = Result;
};

template <size_t N>
using MakeIndexList = typename MakeIndexListT<N>::type;

template <typename ValueList>
struct ReverseT;
template <typename T>
struct ReverseT<ValueList<T>> { using type = ValueList<T>; };

template <typename T, unsigned I1, unsigned... Indices>
struct ReverseT<ValueList<T, I1, Indices...>> {
    /// 1 2 3 => 1 (2 3) => (3 2) 1
    using type = typename PushBackT<
                    typename ReverseT<ValueList<T, Indices...>>::type, 
                        CTValue<T, I1>>::type;
};

template <typename... Es, unsigned... Indices>
auto reverseImpl(const Tuple<Es...>& t, ValueList<unsigned, Indices...>)
{
    return makeTuple(get<Indices>(t)...);
}

template <typename... Ts>
auto reverseTuple(const Tuple<Ts...>& ts)
{
    using indices_t = typename ReverseT<MakeIndexList<sizeof...(Ts)>>::type;
    return reverseImpl(ts, indices_t());                                                                                                                                                                                
}

int main()
{
    Tuple<int, double, std::string> t(10, 20.3, "hello world");
    std::cout << "head: " << t.getHead() << "\n";
    std::cout << "0: " << get<0>(t) << "\n";
    std::cout << "1: " << get<1>(t) << "\n";
    std::cout << "2: " << get<2>(t) << "\n";

    Tuple<double, double, std::string> t1(t);
    std::cout << "head: " << t1.getHead() << "\n";
    std::cout << "0: " << get<0>(t1) << "\n";
    std::cout << "1: " << get<1>(t1) << "\n";
    std::cout << "2: " << get<2>(t1) << "\n";

    std::cout << t1 << "\n";

    static_assert(std::is_same_v<
                Reverse_t<decltype(t)>,
                Tuple<std::string, double, int>>);
    static_assert(std::is_same_v<
                PopFront_t<decltype(t)>,
                Tuple<double, std::string>>);
    static_assert(std::is_same_v<
                PopBack_t<decltype(t)>,
                Tuple<int, double>>);
    static_assert(std::is_same_v<
                PushFront_t<decltype(t), float>,
                Tuple<float, int, double, std::string>>);
    static_assert(std::is_same_v<
                PushBack_t<decltype(t), float>,
                Tuple<int, double, std::string, float>>);

    auto t3 = makeTuple(1, 20, 30.f, 40.2, "hello world");
    std::cout << "t3 = " << t3 << "\n";

    auto t4 = reverseTuple(t3);
    std::cout << "t4 = " << t4 << "\n";
}
```c++
#include <iostream>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iterator>

template <typename...>
struct MatchOverloads;

template <>
struct MatchOverloads<> {
    static void match(...);  // just declaration, no implementation
};

template <typename T, typename... Rest>
struct MatchOverloads<T, Rest...> : public MatchOverloads<Rest...>
{
    /// 根据参数类型T overload来决定最好的match
    static T match(T);  // just declaration, no implementation
    using MatchOverloads<Rest...>::match;
};

template <typename T, typename... Types>
struct BestInMatchTHelper {
    using Type = decltype(MatchOverloads<Types...>::match(std::declval<T>()));
};

template <typename T, typename... Types>
using BestInMatchT = typename BestInMatchTHelper<T, Types...>::Type;

template <typename T,
    typename = BestInMatchT<
        typename std::iterator_traits<T>::iterator_category,
        std::input_iterator_tag,
        std::bidirectional_iterator_tag,
        std::random_access_iterator_tag
    >
> struct Advance;

template <typename T>
struct Advance<T, std::input_iterator_tag>
{
    using distance_t = 
        typename std::iterator_traits<T>::difference_type;
    void operator ()(T& iter, distance_t dist) const {
        while (dist-- > 0) {
            ++iter;
        }
    }
};

template <typename T>
struct Advance<T, std::bidirectional_iterator_tag>
{
    using distance_t = 
        typename std::iterator_traits<T>::difference_type;
    void operator ()(T& iter, distance_t dist) const {
        if (dist > 0) {
            while (dist-- > 0) {
                ++iter;
            }
        } else {
            while (dist++ < 0) {
                --iter;
            }
        }
    }
};

template <typename T>
struct Advance<T, std::random_access_iterator_tag>
{
    using distance_t = 
        typename std::iterator_traits<T>::difference_type;
    void operator ()(T& iter, distance_t dist) const {
        iter += dist;
    }
};

template <typename Iter>
void advance(Iter& iter, int n)
{
    Advance<Iter> ad;
    ad(iter, n);
}

int main()
{
    std::vector<int> a{1,2,3,4,5};
    auto iter = a.begin();
    advance(iter, 1);
    std::cout << *iter << "\n";
}

```

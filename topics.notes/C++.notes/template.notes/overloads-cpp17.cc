#include <iostream>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iterator>
#include <memory>
#include <cstring>
#include <cassert>

template <typename...>
struct MatchOverloads;

template <>
struct MatchOverloads<> {
    static void match(...);  // just declaration, no implementation
};

template <typename T, typename... Rest>
struct MatchOverloads<T, Rest...> : public MatchOverloads<Rest...>
{
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

template <typename... Ts>
struct Overloads : Ts...
{
    using Ts::operator ()...;
};

template <typename... Ts>
Overloads(Ts...) -> Overloads<Ts...>;

template <typename... Args>
auto sum1(Args... args)
{
    //return (1 + ... + args);
    int sum = 0;
    ((sum += args, sum), ...);
    return sum;
}

template <typename... Args>
void print(Args... args)
{
    (std::cout << ... << args) << "\n";
    const char* sep = "";
    (((std::cout << sep << args), sep = ", ", 0), ...);
}

struct MyStruct {
    int GetX() const { return x; }
    const std::string& GetY() const { return y; }
private:
    int x{100};
    std::string y{"yvalue"};
    int z{300};
};

MyStruct GetStruct() {
    return MyStruct{};
};

template <size_t Idx>
auto get(const MyStruct& t) {
    if constexpr (Idx == 0) {
        return t.GetX();
    } else if constexpr (Idx == 1) {
        return t.GetY();
    }
}
namespace std {
    template <> struct tuple_size<MyStruct> : std::integral_constant<size_t, 2> {};
    template <> struct tuple_element<0, MyStruct> { using type = int; };
    template <> struct tuple_element<1, MyStruct> { using type = std::string; };
}

template <typename T>
struct DeleteByOperator {
    void operator ()(T* ptr) {
        delete ptr;
    }
};

template <typename T,
    template <typename> class DeletionPolicy = 
        DeleteByOperator>
struct SmartPointer {
    SmartPointer(T* ptr,
        const DeletionPolicy<T>& deletion_policy = DeletionPolicy<T>())
        : ptr_(ptr)
        , deletion_policy_(deletion_policy)
    {
    }
    
private:
    T* ptr_;
    DeletionPolicy<T> deletion_policy_;
};

struct MyHeap { void free(void* ptr) { } };

template <typename T>
struct DeleteHeap {
    /*explicit*/ DeleteHeap(MyHeap& heap)
        : heap_(heap)
    {}
    void operator ()(T* ptr) {
        heap_.free(ptr);
    }
private:
    MyHeap& heap_;
};

struct SimpleStr {
public:
    SimpleStr() = default;
    SimpleStr(const char* s)
    {
        if (strlen(s) <= sizeof(b_.buf_)) {
            strcpy(b_.buf_, s);
            settag(0);
        } else {
            s_ = strdup(s);
            settag(1);
        }
    }
    SimpleStr(const SimpleStr& s)
    {
        if (s.islocal()) {
            strcpy(b_.buf_, s.b_.buf_);
            settag(0);
        } else {
            s_ = strdup(s.s_);
            settag(1);
        }
    }

    SimpleStr& operator =(const SimpleStr& s)
    {
        if (this == &s) return *this;
        destroy();

        if (s.islocal()) {
            strcpy(b_.buf_, s.b_.buf_);
            settag(0);
        } else {
            s_ = strdup(s.s_);
            settag(1);
        }
        return *this;
    }

    SimpleStr& operator =(const char* s)
    {
        destroy();
        if (strlen(s) <= sizeof(b_.buf_)) {
            strcpy(b_.buf_, s);
            settag(0);
        } else {
            s_ = strdup(s);
            settag(1);
        }
        return *this;
    }

    SimpleStr(SimpleStr&& s)
    {
        if (s.islocal()) {
            strcpy(b_.buf_, s.b_.buf_);
            settag(0);
        } else {
            s_ = s.s_;
            settag(1);
        }
        s.reset();
    }
    SimpleStr& operator =(SimpleStr&& s)
    {
        if (this == &s) return *this;
        destroy();
        if (s.islocal()) {
            strcpy(b_.buf_, s.b_.buf_);
            settag(0);
        } else {
            s_ = s.s_;
            settag(1);
        }
        s.reset();
        return *this;
    }

    ~SimpleStr()
    {
        destroy();
    }

    const char* c_str() const {
        return islocal() ? b_.buf_ : s_;
    }
    friend std::ostream& operator <<(std::ostream& os, const SimpleStr& ss)
    {
        return os << ss.c_str();
    }
    friend bool operator ==(const SimpleStr& s1, const SimpleStr& s2)
    {
        return strcmp(s1.c_str(), s2.c_str()) == 0;
    }
    friend bool operator !=(const SimpleStr& s1, const SimpleStr& s2)
    {
        return !(s1 == s2);
    }
private:
    void destroy()
    {
        if (!islocal()) {
            free(s_);
        }
    }
    void reset()
    {
        b_.buf_[0] = '\0';
        settag(0);
    }
    bool islocal() const {
        return b_.tag_ == 0;
    }
    void settag(int tag) {
        b_.tag_ = tag;
    }
private:
    union {
        char* s_;
        struct {
            char buf_[15];
            char tag_;
            // tag=0, s == buf, local
            // tag=1, s != buf, external
        } b_;
    };
};

int main()
{
    SimpleStr ss;
    std::cout << ss << "\n";
    {
        SimpleStr s1("helloworld");
        std::cout << s1 << "\n";
        SimpleStr s2("helloworld");
        assert(s1 == s2);

        auto s3 = s1;
        assert(s1 == s3);
        auto s4(s1);
        assert(s4 == s1);
        s1 = "go to";
        std::cout << s1 << "\n";
        assert(s1 != s4);

        s1 = std::move(s4);
        std::cout << s1 << "\n";
        std::cout << s4 << "\n";
        
        auto s5(std::move(s1));
        std::cout << "s5: " << s5 << "\n";
        std::cout << "s1: " << s1 << "\n";
    }
}


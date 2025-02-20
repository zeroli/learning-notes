#include <iostream>
#include <string>
#include <type_traits>
#include <cassert>
#include <utility>

/// abstract Ptr<T> and Ref<T> for pointer and reference
template <typename T>
struct Ref;

template <typename T>
struct Ptr {
public:
    using value_type = T;
    using different_type = std::ptrdiff_t;
    using pointer_type = Ptr;
    using reference_type = Ref<T>;
    using const_reference_type = Ref<std::add_const_t<T>>;

    Ptr() = default;
    /// templated to allow copy from T* to const T*
    template <typename Y>
    /*explicit*/ Ptr(Y* ptr)
        : ptr_(ptr)
    {}
    ~Ptr() = default;

    /// copy support from another pointer with different type
    template <typename U>
    Ptr(const Ptr<U>& u)
        : ptr_(u.get())
    {}
    template <typename U>
    Ptr& operator =(const Ptr<U>& u)
    {
        ptr_ = u.get();
        return *this;
    }

    /// move support
    template <typename U>
    Ptr(Ptr<U>&& u)
        : ptr_(std::exchange(u.ptr_, nullptr))
    {}
    template <typename U>
    Ptr& operator =(Ptr<U>&& u)
    {
        ptr_ = std::exchange(u.ptr_, nullptr);
        return *this;
    }

    /// reset to null
    Ptr(std::nullptr_t) : ptr_(nullptr) {}
    Ptr& operator =(std::nullptr_t) {
        ptr_ = nullptr;
        return *this;
    }

    /// deference
    reference_type operator *() {
        assert(ptr_);
        return reference_type(*this);
    }
    const_reference_type& operator *() const {
        assert(ptr_);
        return const_reference_type(*this);
    }

    T* operator ->() { return ptr_; }
    const T* operator ->() const { return ptr_; }

    T* get() const { return ptr_; }

    explicit operator bool() const {
        return ptr_ != nullptr;
    }

    reference_type operator [](different_type diff) {
        return reference_type(Ptr(ptr_ + diff));
    }
    const_reference_type operator [](different_type diff) const {
        return const_reference_type(Ptr(ptr_ + diff));
    }

    /// pointer arithmetic operations
    Ptr& operator ++() {
        ++ptr_;
        return *this;
    }
    Ptr operator ++(int) {
        Ptr old(*this);
        ++ptr_;
        return old;
    }
    Ptr& operator --() {
        --ptr_;
        return *this;
    }
    Ptr operator --(int) {
        Ptr old(*this);
        --ptr_;
        return old;
    }

    Ptr& operator +=(different_type diff) {
        ptr_ += diff;
        return *this;
    }
    Ptr& operator -=(different_type diff) {
        ptr_ -= diff;
        return *this;
    }

    Ptr operator +(int diff) const {
        return Ptr(ptr_ + diff);
    }
    Ptr operator -(int diff) const {
        return Ptr(ptr_ - diff);
    }

    friend different_type operator -(const Ptr& p1, const Ptr& p2) {
        return difference_type(p1.ptr_ - p2.ptr_);
    }
    friend bool operator ==(const Ptr& p1, const Ptr& p2)
    {
        return p1.ptr_ == p2.ptr_;
    }
    friend bool operator !=(const Ptr& p1, const Ptr& p2)
    {
        return !(p1 == p2);
    }
    friend std::ostream& operator <<(std::ostream& os, const Ptr& p)
    {
        if (p) os << p.ptr_;
        else os << "(null)";
        return os;
    }
private:
    T* ptr_{nullptr};
};

template <typename T>
struct Ref {
public:
    using pointer_type = Ptr<T>;

    explicit Ref(const pointer_type& ptr)
        : ptr_(ptr)
    {}

    ~Ref() = default;

    Ref(const Ref& ref)
        : ptr_(ref.ptr_)
    {}

    Ref& operator =(const Ref& ref) = delete;

    /// addressable
    pointer_type operator &() const {
        return ptr_;
    }
    /// assignment from one value
    Ref& operator =(const T& val)
    {
        *ptr_.get() = val;
        return *this;
    }
    /// fetch its value
    operator T() const {
        assert(ptr_);
        return *ptr_.get();
    }

    Ref& operator ++() {
        auto& v = *ptr_.get();
        ++v;
        return *this;
    }
    T operator ++(int) {
        auto& v = *ptr_.get();
        T old = v++;
        return old;
    }
    Ref& operator --() {
        auto& v = *ptr_.get();
        --v;
        return *this;
    }
    T operator --(int) {
        auto& v = *ptr_.get();
        T old = v--;
        return old;
    }

    friend std::ostream& operator <<(std::ostream& os, const Ref& r)
    {
        return os << (T)r;
    }
private:
    pointer_type ptr_;
};

int main()
{
    Ptr<int> p1;
    std::cout << p1 << "\n";
    {
        Ptr<int> p2(p1);
        std::cout << p2 << "\n";
        p2 = nullptr;
        std::cout << p2 << "\n";
        Ptr<int> p(new int(3));
        std::cout << p << "\n";
        std::cout << *p << "\n";
        Ptr<int> p3(p);
        std::cout << p3 << "\n";
        std::cout << *p3 << "\n";
        p3++; --p3;
        std::cout << *p3 << "\n";
        auto p4(std::move(p3));
        std::cout << p3 << "\n";
        std::cout << *p4 << "\n";
    }
    {
        Ptr<int> p(new int(10));
        std::cout << *p << "\n";
        Ref<int> ref(p);
        ++ref;
        std::cout << *p << "\n";
        std::cout << ref << "\n";
    }
    {
        Ptr<int> p = new int[10]{};
        p[0] = 10;
        p[1] = 20;
        p[3] = 30;
        for (int i = 0; i < 10; i++) {
           if (i != 0) {
                std::cout << ", ";
            }
            std::cout << p[i];
        }
        Ptr<void> vp(p);
        Ptr<void> vp1;
        vp1 = p;
    }
}

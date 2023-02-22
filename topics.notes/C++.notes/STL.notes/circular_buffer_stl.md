# 实现一个circurlar_buffer/circular_buffer_iterator
```c++
#include <iostream>
#include <iterator>
#include <array>
#include <algorithm>
#include <sstream>
#include <cassert>

using namespace std;

template <typename T, size_t N>
struct circular_buffer_iterator;

template <typename T, size_t N>
struct circular_buffer {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using reference = value_type&;
    using const_reference = value_type const&;

    using iterator = circular_buffer_iterator<T, N>;
    using const_iterator = circular_buffer_iterator<const T, N>;
    
    friend struct circular_buffer_iterator<T, N>;
public:
    circular_buffer() = default;
    circular_buffer(value_type const (&array)[N])
        : size_(N), tail_(N-1)
    {
        std::copy(std::begin(array), std::end(array), data_.begin());
    }
    circular_buffer(value_type const& t)
        : size_(N), tail_(N-1)
    {
        std::fill(std::begin(data_), std::end(data_), t);
    }
    
    size_t capacity() const { return N; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == N; }
    
    void push_back(value_type const& t) {
        if (empty()) {
            data_[tail_] = t;
            size_++;
        } else if (!full()) {
            data_[++tail_] = t;
            size_++;
        } else {
            tail_ = (tail_ + 1) % capacity();
            head_ = (head_ + 1) % capacity();
            data_[tail_] = t;
        }
    }
    
    value_type pop_front() {
        if (empty()) {
            throw std::logic_error("buffer is empty");
        }
        value_type val = data_[head_];
        head_ = (head_ + 1) % capacity();
        return val;
    }
    
    reference operator[](size_t index) {
        if (index >= size_) {
            throw std::logic_error("index out of range");
        }
        return data_[(head_ + index) % capacity()];
    }
    const_reference operator[](size_t index) const {
        if (index >= size_) {
            throw std::logic_error("index out of range");
        }
        return data_[(head_ + index) % capacity()];
    }
    
    reference front() {
        if (empty()) {
            throw std::logic_error("buffer is empty");
        }
        return data_[head_];
    }
    reference back() {
        if (empty()) {
            throw std::logic_error("buffer is empty");
        }
        return data_[tail_];
    }

    const_reference front() const {
        if (empty()) {
            throw std::logic_error("buffer is empty");
        }
        return data_[head_];
    }
    const_reference back() const {
        if (empty()) {
            throw std::logic_error("buffer is empty");
        }
        return data_[tail_];
    }

    iterator begin() {
        return iterator(*this, 0);
    }
    
    iterator end() {
        return iterator(*this, capacity());
    }
    
    const_iterator begin() const {
        return const_iterator(*this, 0);
    }
    
    const_iterator end() const {
        return const_iterator(*this, capacity());
    }
    
    friend std::ostream& operator <<(std::ostream& os, circular_buffer& buffer) {
        std::ostringstream outstr;
        outstr << "[";
        for (size_t i = 0; i < buffer.size_; i++) {
            outstr << buffer[i] << ", ";
        }
        if (buffer.size_ > 0) {
            outstr.seekp(-2, std::ios_base::cur);
        }
        outstr << "]";
        return os << outstr.str();
    }
private:
    std::array<T, N> data_;
    size_t size_ = 0;
    size_t head_ = 0;
    size_t tail_ = 0;
};

template <typename T, size_t N>
struct circular_buffer_iterator
{
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using reference = value_type&;
    using const_reference = value_type const&;
    
    using self_type = circular_buffer_iterator<T, N>;
public:
    circular_buffer_iterator(circular_buffer<T, N>& buffer, size_t index)
        : buffer_(buffer), index_(index)
    { }
    
    
    self_type& operator ++() {
        if (index_ >= buffer_.get().size()) {
            throw std::logic_error("index out of range");
        }
        index_++;
        return *this;
    }
    self_type operator ++(int) {
        self_type tmp = *this;
        ++*this;
        return tmp;
    }
    
    self_type& operator --() {
        if (index_ <= 0) {
            throw std::logic_error("index out of range");
        }
        index_--;
        return *this;
    }
    self_type operator --(int) {
        self_type tmp = *this;
        ++*this;
        return tmp;
    }

    self_type operator +(difference_type n) {
        self_type tmp = *this;
        return tmp += n;
    }
    self_type operator -(difference_type n) {
        self_type tmp = *this;
        return tmp -= n;
    }
    
    self_type& operator +=(difference_type n) {
        auto next = (index_ + n) % buffer_.get().capacity();
        if (next > buffer_.get().size()) {
            throw std::logic_error("index out of range");
        }
        index_ = next;
        return *this;
    }
    self_type& operator -=(difference_type n) {
        return operator += (-n);
    }

    const_reference operator *() const {
        if (index_ >= buffer_.get().size()) {
            throw std::logic_error("index out of range");
        }
        auto& buf = buffer_.get();
        return buf.data_[(buf.head_ + index_) % buffer_.get().capacity()];
    }
    
    const_reference operator ->() const {
        if (index_ >= buffer_.get().size()) {
            throw std::logic_error("index out of range");
        }
        auto& buf = buffer_.get();
        return buf.data_[(buf.head_ + index_) % buffer_.get().capacity()];
    }
    reference operator *() {
        if (index_ >= buffer_.get().size()) {
            throw std::logic_error("index out of range");
        }
        auto& buf = buffer_.get();
        return buf.data_[(buf.head_ + index_) % buffer_.get().capacity()];
    }
    
    reference operator ->() {
        if (index_ >= buffer_.get().size()) {
            throw std::logic_error("index out of range");
        }
        auto& buf = buffer_.get();
        return buf.data_[(buf.head_ + index_) % buffer_.get().capacity()];
    }
    
    bool operator ==(const self_type& other) const {
        return &(buffer_.get().data_) == &(other.buffer_.get().data_) && (index_ == other.index_);
    }
    bool operator !=(const self_type& other) const {
        return !(*this == other);
    }
    
    
private:
    std::reference_wrapper<circular_buffer<T, N>> buffer_;
    size_t index_;
};

int main()
{
    {
        circular_buffer<int, 1> b1;
        b1.push_back(10);
        std::cout << b1 << "\n";
        assert(b1[0] = 10);
        assert(b1.front() == 10);
    }
    {
        circular_buffer<int, 3> b1({10, 20, 30});
        b1[1] = 30;
        std::cout << b1 << "\n";
        assert(b1[1] = 30);
        assert(b1.front() == 10);
        assert(b1.back() == 30);
    }
    {
        circular_buffer<int, 3> b1({10, 20, 30});
        for (auto& v : b1) {
            v = v * 2;
        }
        std::cout << b1 << "\n";
    }
    return 0;
}
```

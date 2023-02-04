```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;

template <typename T, typename C = std::vector<T>>
class Vector {
public:
    Vector() = default;
    
    Vector(const C& c) : data_(c) { }
    
    Vector(std::initializer_list<T> t)
        : data_(t)
    {
    }
    
    template <typename U, typename X>
    Vector(const Vector<U, X>& v) {
        data_.resize(v.size());
        for (size_t i = 0u; i < v.size(); i++) {
            data_[i] = v.data_[i];
        }
    }
    template <typename U, typename X>
    Vector& operator=(const Vector<U, X>& v) {
        data_.resize(v.size());
        for (size_t i = 0u; i < v.size(); i++) {
            data_[i] = v.data_[i];
        }
        return *this;
    }
    
    //T& operator[](size_t i) { return data_[i]; }
    T operator[](size_t i) const { return data_[i]; }
    size_t size() const { return data_.size(); }
    
    C& data() { return data_; }
    const C& data() const { return data_; }
private:
    C data_;
};

template <typename L, typename R>
class VectorAdd {
public:
    VectorAdd(const L& l, const R& r)
        : l(l), r(r)
    { }
    
    auto operator[](size_t i) const {
        return l[i] + r[i];
    }
    
    size_t size() const {
        return l.size();
    }
private:
    const L& l;
    const R& r;
};

template <typename T1, typename C1, typename T2, typename C2>
auto operator +(const Vector<T1, C1>& v1, const Vector<T2, C2>& v2)
{
    using result_t = decltype(std::declval<T1>() + std::declval<T2>());
    return Vector<result_t, VectorAdd<C1, C2>>{
            VectorAdd<C1, C2>{v1.data(), v2.data()}};
}

int main()
{
    Vector<int> v1 = { 1, 2, 3 };
    Vector<int> v2 = { 2, 3, 4 };
    auto v3 = v1 + v2 + v1;
    for (int i = 0; i < 3; i++) {
        std::cout << v3[i] << ",";
    }
    
    return 0;
}
```


产生的汇编代码如下：
```asm
Vector<int, std::vector<int, std::allocator<int> > >::~Vector() [base object destructor]:
  pushq %rbp
  movq %rsp, %rbp
  subq $16, %rsp
  movq %rdi, -8(%rbp)
  movq -8(%rbp), %rax
  movq %rax, %rdi
  call std::vector<int, std::allocator<int> >::~vector() [complete object destructor]
  nop
  leave
  ret
auto operator+<int, std::vector<int, std::allocator<int> >, int, std::vector<int, std::allocator<int> > >(Vector<int, std::vector<int, std::allocator<int> > > const&, Vector<int, std::vector<int, std::allocator<int> > > const&):
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  subq $56, %rsp
  movq %rdi, -56(%rbp)
  movq %rsi, -64(%rbp)
  movq -56(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::data() const
  movq %rax, %rbx
  movq -64(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::data() const
  movq %rax, %rdx
  leaq -32(%rbp), %rax
  movq %rbx, %rsi
  movq %rax, %rdi
  call VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >::VectorAdd(std::vector<int, std::allocator<int> > const&, std::vector<int, std::allocator<int> > const&) [complete object constructor]
  leaq -32(%rbp), %rdx
  leaq -48(%rbp), %rax
  movq %rdx, %rsi
  movq %rax, %rdi
  call Vector<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > >::Vector(VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > const&) [complete object constructor]
  movq -48(%rbp), %rax
  movq -40(%rbp), %rdx
  movq -8(%rbp), %rbx
  leave
  ret
auto operator+<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, int, std::vector<int, std::allocator<int> > >(Vector<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > > const&, Vector<int, std::vector<int, std::allocator<int> > > const&):
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  subq $56, %rsp
  movq %rdi, -56(%rbp)
  movq %rsi, -64(%rbp)
  movq -56(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > >::data() const
  movq %rax, %rbx
  movq -64(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::data() const
  movq %rax, %rdx
  leaq -32(%rbp), %rax
  movq %rbx, %rsi
  movq %rax, %rdi
  call VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > >::VectorAdd(VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > const&, std::vector<int, std::allocator<int> > const&) [complete object constructor]
  leaq -32(%rbp), %rdx
  leaq -48(%rbp), %rax
  movq %rdx, %rsi
  movq %rax, %rdi
  call Vector<int, VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > > >::Vector(VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > > const&) [complete object constructor]
  movq -48(%rbp), %rax
  movq -40(%rbp), %rdx
  movq -8(%rbp), %rbx
  leave
  ret
.LC0:
  .string ","
main:
  pushq %rbp
  movq %rsp, %rbp
  pushq %r13
  pushq %r12
  pushq %rbx
  subq $136, %rsp
  movl $1, -88(%rbp)
  movl $2, -84(%rbp)
  movl $3, -80(%rbp)
  leaq -88(%rbp), %rcx
  movq %rcx, %rax
  movl $3, %edx
  movq %rax, %rcx
  movq %rdx, %rbx
  leaq -112(%rbp), %rax
  movq %rcx, %rsi
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::Vector(std::initializer_list<int>) [complete object constructor]
  movl $2, -76(%rbp)
  movl $3, -72(%rbp)
  movl $4, -68(%rbp)
  leaq -76(%rbp), %rax
  movq %rax, %r12
  movl $3, %r13d
  movq %r12, %rcx
  movq %r13, %rbx
  movq %r12, %rax
  movq %r13, %rdx
  leaq -144(%rbp), %rax
  movq %rcx, %rsi
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::Vector(std::initializer_list<int>) [complete object constructor]
  leaq -144(%rbp), %rdx
  leaq -112(%rbp), %rax
  movq %rdx, %rsi
  movq %rax, %rdi
  call auto operator+<int, std::vector<int, std::allocator<int> >, int, std::vector<int, std::allocator<int> > >(Vector<int, std::vector<int, std::allocator<int> > > const&, Vector<int, std::vector<int, std::allocator<int> > > const&)
  movq %rax, -64(%rbp)
  movq %rdx, -56(%rbp)
  leaq -112(%rbp), %rdx
  leaq -64(%rbp), %rax
  movq %rdx, %rsi
  movq %rax, %rdi
  call auto operator+<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, int, std::vector<int, std::allocator<int> > >(Vector<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > > const&, Vector<int, std::vector<int, std::allocator<int> > > const&)
  movq %rax, -160(%rbp)
  movq %rdx, -152(%rbp)
  movl $0, -36(%rbp)
  jmp .L7
.L8:
  movl -36(%rbp), %eax
  movslq %eax, %rdx
  leaq -160(%rbp), %rax
  movq %rdx, %rsi
  movq %rax, %rdi
  call Vector<int, VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > > >::operator[](unsigned long) const
  movl %eax, %esi
  movl $_ZSt4cout, %edi
  call std::basic_ostream<char, std::char_traits<char> >::operator<<(int)
  movl $.LC0, %esi
  movq %rax, %rdi
  call std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)
  addl $1, -36(%rbp)
.L7:
  cmpl $2, -36(%rbp)
  jle .L8
  movl $0, %ebx
  leaq -144(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::~Vector() [complete object destructor]
  leaq -112(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::~Vector() [complete object destructor]
  movl %ebx, %eax
  jmp .L14
  movq %rax, %rbx
  leaq -144(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::~Vector() [complete object destructor]
  jmp .L11
  movq %rax, %rbx
.L11:
  leaq -112(%rbp), %rax
  movq %rax, %rdi
  call Vector<int, std::vector<int, std::allocator<int> > >::~Vector() [complete object destructor]
  movq %rbx, %rax
  movq %rax, %rdi
  call _Unwind_Resume
.L14:
  addq $136, %rsp
  popq %rbx
  popq %r12
  popq %r13
  popq %rbp
  ret
Vector<int, std::vector<int, std::allocator<int> > >::Vector(std::initializer_list<int>) [base object constructor]:
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  subq $56, %rsp
  movq %rdi, -40(%rbp)
  movq %rdx, %rcx
  movq %rsi, %rax
  movq %rdi, %rdx
  movq %rcx, %rdx
  movq %rax, -64(%rbp)
  movq %rdx, -56(%rbp)
  movq -40(%rbp), %rbx
  leaq -17(%rbp), %rax
  movq %rax, %rdi
  call std::allocator<int>::allocator() [complete object constructor]
  leaq -17(%rbp), %rcx
  movq -64(%rbp), %rdx
  movq -56(%rbp), %rax
  movq %rdx, %rsi
  movq %rax, %rdx
  movq %rbx, %rdi
  call std::vector<int, std::allocator<int> >::vector(std::initializer_list<int>, std::allocator<int> const&) [complete object constructor]
  leaq -17(%rbp), %rax
  movq %rax, %rdi
  call std::allocator<int>::~allocator() [complete object destructor]
  jmp .L18
  movq %rax, %rbx
  leaq -17(%rbp), %rax
  movq %rax, %rdi
  call std::allocator<int>::~allocator() [complete object destructor]
  movq %rbx, %rax
  movq %rax, %rdi
  call _Unwind_Resume
.L18:
  movq -8(%rbp), %rbx
  leave
  ret
Vector<int, std::vector<int, std::allocator<int> > >::data() const:
  pushq %rbp
  movq %rsp, %rbp
  movq %rdi, -8(%rbp)
  movq -8(%rbp), %rax
  popq %rbp
  ret
VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >::VectorAdd(std::vector<int, std::allocator<int> > const&, std::vector<int, std::allocator<int> > const&) [base object constructor]:
  pushq %rbp
  movq %rsp, %rbp
  movq %rdi, -8(%rbp)
  movq %rsi, -16(%rbp)
  movq %rdx, -24(%rbp)
  movq -8(%rbp), %rax
  movq -16(%rbp), %rdx
  movq %rdx, (%rax)
  movq -8(%rbp), %rax
  movq -24(%rbp), %rdx
  movq %rdx, 8(%rax)
  nop
  popq %rbp
  ret
Vector<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > >::Vector(VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > const&) [base object constructor]:
  pushq %rbp
  movq %rsp, %rbp
  movq %rdi, -8(%rbp)
  movq %rsi, -16(%rbp)
  movq -8(%rbp), %rcx
  movq -16(%rbp), %rax
  movq 8(%rax), %rdx
  movq (%rax), %rax
  movq %rax, (%rcx)
  movq %rdx, 8(%rcx)
  nop
  popq %rbp
  ret
Vector<int, VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > >::data() const:
  pushq %rbp
  movq %rsp, %rbp
  movq %rdi, -8(%rbp)
  movq -8(%rbp), %rax
  popq %rbp
  ret
VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > >::VectorAdd(VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > const&, std::vector<int, std::allocator<int> > const&) [base object constructor]:
  pushq %rbp
  movq %rsp, %rbp
  movq %rdi, -8(%rbp)
  movq %rsi, -16(%rbp)
  movq %rdx, -24(%rbp)
  movq -8(%rbp), %rax
  movq -16(%rbp), %rdx
  movq %rdx, (%rax)
  movq -8(%rbp), %rax
  movq -24(%rbp), %rdx
  movq %rdx, 8(%rax)
  nop
  popq %rbp
  ret
Vector<int, VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > > >::Vector(VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > > const&) [base object constructor]:
  pushq %rbp
  movq %rsp, %rbp
  movq %rdi, -8(%rbp)
  movq %rsi, -16(%rbp)
  movq -8(%rbp), %rcx
  movq -16(%rbp), %rax
  movq 8(%rax), %rdx
  movq (%rax), %rax
  movq %rax, (%rcx)
  movq %rdx, 8(%rcx)
  nop
  popq %rbp
  ret
VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >::operator[](unsigned long) const:
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  subq $24, %rsp
  movq %rdi, -24(%rbp)
  movq %rsi, -32(%rbp)
  movq -24(%rbp), %rax
  movq (%rax), %rax
  movq -32(%rbp), %rdx
  movq %rdx, %rsi
  movq %rax, %rdi
  call std::vector<int, std::allocator<int> >::operator[](unsigned long) const
  movl (%rax), %ebx
  movq -24(%rbp), %rax
  movq 8(%rax), %rax
  movq -32(%rbp), %rdx
  movq %rdx, %rsi
  movq %rax, %rdi
  call std::vector<int, std::allocator<int> >::operator[](unsigned long) const
  movl (%rax), %eax
  addl %ebx, %eax
  movq -8(%rbp), %rbx
  leave
  ret
VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > >::operator[](unsigned long) const:
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  subq $24, %rsp
  movq %rdi, -24(%rbp)
  movq %rsi, -32(%rbp)
  movq -24(%rbp), %rax
  movq (%rax), %rax
  movq -32(%rbp), %rdx
  movq %rdx, %rsi
  movq %rax, %rdi
  call VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >::operator[](unsigned long) const
  movl %eax, %ebx
  movq -24(%rbp), %rax
  movq 8(%rax), %rax
  movq -32(%rbp), %rdx
  movq %rdx, %rsi
  movq %rax, %rdi
  call std::vector<int, std::allocator<int> >::operator[](unsigned long) const
  movl (%rax), %eax
  addl %ebx, %eax
  movq -8(%rbp), %rbx
  leave
  ret
Vector<int, VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > > >::operator[](unsigned long) const:
  pushq %rbp
  movq %rsp, %rbp
  subq $16, %rsp
  movq %rdi, -8(%rbp)
  movq %rsi, -16(%rbp)
  movq -8(%rbp), %rax
  movq -16(%rbp), %rdx
  movq %rdx, %rsi
  movq %rax, %rdi
  call VectorAdd<VectorAdd<std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::vector<int, std::allocator<int> > >::operator[](unsigned long) const
  leave
  ret
.LC1:
  .string "cannot create std::vector larger than max_size()"
__static_initialization_and_destruction_0(int, int):
  pushq %rbp
  movq %rsp, %rbp
  subq $16, %rsp
  movl %edi, -4(%rbp)
  movl %esi, -8(%rbp)
  cmpl $1, -4(%rbp)
  jne .L126
  cmpl $65535, -8(%rbp)
  jne .L126
  movl $_ZStL8__ioinit, %edi
  call std::ios_base::Init::Init() [complete object constructor]
  movl $__dso_handle, %edx
  movl $_ZStL8__ioinit, %esi
  movl $_ZNSt8ios_base4InitD1Ev, %edi
  call __cxa_atexit
.L126:
  nop
  leave
  ret
_GLOBAL__sub_I_main:
  pushq %rbp
  movq %rsp, %rbp
  movl $65535, %esi
  movl $1, %edi
  call __static_initialization_and_destruction_0(int, int)
  popq %rbp
  ret
```
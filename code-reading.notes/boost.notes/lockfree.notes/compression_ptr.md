# compression_ptr对指针的封装

类`tagged_ptr`对compression_ptr进行了封装
```c++
#if BOOST_ARCH_X86_64 || defined (__aarch64__)
template <class T>
class tagged_ptr
{
    typedef boost::uint64_t compressed_ptr_t;

public:
    typedef boost::uint16_t tag_t;
```
* 注意到这里compression_ptr只支持x86架构，因为64物理地址指针只有低48位被采用，高16位留作它用。

```c++
    union cast_unit
    {
        compressed_ptr_t value;
        tag_t tag[4];  // 16 * 4 = 64 bits
    };

    static const int tag_index = 3;
    static const compressed_ptr_t ptr_mask = 0xffffffffffffUL; //(1L<<48L)-1;
```
* 高16位被挪用tag只用。

# 三个utility函数：
```c++
    static T* extract_ptr(volatile compressed_ptr_t const & i)
    {
        return (T*)(i & ptr_mask);  // 获取低48位，拿到真正的对象地址指针
    }

    static tag_t extract_tag(volatile compressed_ptr_t const & i)
    {
        cast_unit cu;
        cu.value = i;
        return cu.tag[tag_index];  // 获取高16位，拿到tag
    }

    static compressed_ptr_t pack_ptr(T * ptr, tag_t tag)
    {
        cast_unit ret;
        ret.value = compressed_ptr_t(ptr);
        ret.tag[tag_index] = tag;
        return ret.value;
    }
```

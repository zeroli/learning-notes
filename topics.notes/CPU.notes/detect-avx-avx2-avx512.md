```c++
bool TestAVXSupport(void)
{
    uint32_t ecx;
    constexpr uint32_t probeAVX = 1U << 28;
    __asm__ __volatile__(
        "mov    $1, %%eax \n"
        "xor %%ecx, %%ecx \n"
        "cpuid            \n"
        : "=c"(ecx) :: "eax", "ebx", "edx"
    );
    return (ecx & probeAVX) != 0;
}

bool TestAVX2Support(void)
{
    uint32_t idinfo;
    constexpr uint32_t probeL1 = (3U << 26) | (1U << 22) | (1U << 12);
    constexpr uint32_t probeL7 = (1U << 8) | (1U << 5) | (1U << 3);
    constexpr uint32_t probeL8_1 = (1U << 5);
    __asm__ __volatile__(
        "mov    $1, %%eax \n"
        "xor %%ecx, %%ecx \n"
        "cpuid            \n"
        "mov %%ecx, %0    \n"
        : "=rm"(idinfo) :: "eax", "ebx", "ecx", "edx"
    );
    bool retVal = (idinfo & probeL1) == probeL1;
    __asm__ __volatile__(
        "mov    $7, %%eax \n"
        "xor %%ecx, %%ecx \n"
        "cpuid            \n"
        "mov %%ebx, %0    \n"
        : "=rm"(idinfo) :: "eax", "ebx", "ecx", "edx"
    );
    retVal = retVal && ((idinfo & probeL7) == probeL7);
    __asm__ __volatile__(
        "mov $0x80000001, %%eax \n"
        "xor       %%ecx, %%ecx \n"
        "cpuid                  \n"
        "mov       %%ecx, %0    \n"
        : "=rm"(idinfo) :: "eax", "ebx", "ecx", "edx"
    );
    retVal = retVal && ((idinfo & probeL8_1) == probeL8_1);
    __asm__ __volatile__(
        "xor   %%ecx,   %%ecx \n" //  [0]
        ".byte 0x0f,0x01,0xd0 \n" //  XGETBV
        "and      $6,   %%eax \n" //  xmm and ymm state in XCR0
        "xor      $6,   %%eax \n"
        "mov   %%eax,      %0 \n"
        : "=rm"(idinfo) :: "eax", "ecx", "edx"
    );
    retVal = retVal && (idinfo == 0);

    return retVal;
}

bool TestAVX512FSupport(void)
{
    return false;
}
```

```c++
#ifdef __CUDACC__
__host__ __device__ 
#endif
int countLeadingZeros(unsigned int a)
{
#if defined(__CUDA_ARCH__)
  return __popc(a);
#else
  // Source: http://graphics.stanford.edu/~seander/bithacks.html
  a = a - ((a >> 1) & 0x55555555);                    
  a = (a & 0x33333333) + ((a >> 2) & 0x33333333);     
  return ((a + (a >> 4) & 0xF0F0F0F) * 0x1010101) >> 24; 
#endif
}
```

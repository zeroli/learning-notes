#ifndef COMMON_H_
#define COMMON_H_

#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FLOAT4_w(addr) *(reinterpret_cast<float4*>(addr))
#define FLOAT4_r(addr) *(reinterpret_cast<const float4*>(addr))

#endif  // COMMON_H_

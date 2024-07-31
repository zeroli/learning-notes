```
kernel:  cublas
M N K =   1024   1024   1024, Time =   0.00019917 s, AVG Performance = 10041.7737 Gflops

kernel:  naive
M N K =   1024   1024   1024, Time =   0.00062730 s, AVG Performance =  3188.2550 Gflops

kernel:  smem_block_tile
M N K =   1024   1024   1024, Time =   0.00043008 s, AVG Performance =  4650.2976 Gflops

kernel:  smem_2d_tile
M N K =   1024   1024   1024, Time =   0.00025590 s, AVG Performance =  7815.6261 Gflops

kernel:  smem_2d_tile_float4
M N K =   1024   1024   1024, Time =   0.00024545 s, AVG Performance =  8148.2061 Gflops
```
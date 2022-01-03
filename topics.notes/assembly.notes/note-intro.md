# different basic statement assembly code

while-loop
====
```c++
void while_code()
{
    int i = 1;
    while (i) {
        i--;
    }
}
```
```assembly
while_code():
  pushq %rbp
  movq %rsp, %rbp
  movl $1, -4(%rbp)
  jmp .L2
.L3:
  subl $1, -4(%rbp)
.L2:
  cmpl $0, -4(%rbp)
  jne .L3
  popq %rbp
  ret
```

for-loop
====
```c++
void for_code()
{
    int i = 1;
    int k;
    for (k = 0; k < i; k++) {

    }
}
```
```assembly
for_code():
  pushq %rbp
  movq %rsp, %rbp
  movl $1, -8(%rbp)
  movl $0, -4(%rbp)
  jmp .L5
.L6:
  addl $1, -4(%rbp)
.L5:
  movl -4(%rbp), %eax
  cmpl -8(%rbp), %eax
  jl .L6
  popq %rbp
  ret
```

if/else
====
```c++
void if_else_code()
{
    int i = 1;
    if (i == 0) {
        i = 10;
    } else if (i == 1) {
        i = 20;
    } else {

    }
}
```
```assembly
if_else_code():
  pushq %rbp
  movq %rsp, %rbp
  movl $1, -4(%rbp)
  cmpl $0, -4(%rbp)
  jne .L8
  movl $10, -4(%rbp)
  jmp .L7
.L8:
  cmpl $1, -4(%rbp)
  jne .L7
  movl $20, -4(%rbp)
.L7:
  popq %rbp
  ret
```

switch
====

```c++
void switch_code()
{
    int i = 1;
    switch (i) {
        case 0:
            i = 10;
            break;
        case 1:
            i = 20;
            break;
        default:
            break;
    }
}
```
```assembly
switch_code():
  pushq %rbp
  movq %rsp, %rbp
  movl $1, -4(%rbp)
  movl -4(%rbp), %eax
  testl %eax, %eax
  je .L12
  cmpl $1, %eax
  je .L13
  jmp .L10
.L12:
  movl $10, -4(%rbp)
  jmp .L10
.L13:
  movl $20, -4(%rbp)
  nop
.L10:
  popq %rbp
  ret
```

# 书<揭秘Java虚拟机-JVM设计原理与实现>阅读笔记

## 第2章：Java执行引擎工作原理: 方法调用
=======
* 方法调用
类似于C/C++里面的函数调用，是程序组成的基本单元，作为原子指令的初步封装。
Java语言的源自指令是字节码，Java方法是对字节码的封装。
* 取指
取出指令，逐条去除字节码指令。
* 运算

Java程序最基本的组成单位是类，而Java类也是由一个个的函数所组成。
Java VM作为一款虚拟机，想要具备执行一个完整的Java程序的能力，就必定得具备执行单个Java函数的能力，而要具备执行Java函数的能力，首先必须得能执行函数调用。

Java VM也是将Java函数所对应得机器指令专门存储在内存的一块区域上，同时为每一个Java函数分配方法栈。

main函数其实只需要分配4字节的堆栈空间，用于保存计算结果3，可为啥编译器为其分配16字节的空间呢？这是一种约定，就是内存对齐。在32位和64位机器上，堆栈内存都是按照16字节对齐的，多了不退，少了一定会补齐。道理很简单，就是为了能够对内存进行快速定位、快速整理回收。
```c++
int add();
int main() {
    int c = add();
    return 0;
}
int add() {
    int z = 1 + 2;
    return z;
}
```
```asm
main:
  pushl %ebp
  movl %esp, %ebp
  andl %-16, %esp  # align %esp to 16
  subl %16, %esp  # allocate 16 byte down for stack, %esp - 16

  call add
  mov %0，%eax
  leave  # => movl %ebp，%esp; pop %ebp
  ret

add:
  pushl %ebp
  movl %esp, %ebp
  subl %16, %esp
  mov %3, -4(%ebp)
  movl -4(%esp), %eax
  leave
  ret
```

带参数的add函数C/C++
```c++
int add(int a, int b);
int main() {
    int a = 5;
    int b = 3;
    int c = add(a, b);
    return 0;
}
int add(int a, int b) {
    int z = a + b;
    return z;
}
```
```asm
main:
  pushl %ebp
  movl %esp, %ebp
  andl %-16, %esp  # align %esp to 16
  subl %16, %esp  # allocate 16 byte down for stack, %esp - 16

  call add
  mov %0，%eax
  leave  # => movl %ebp，%esp; pop %ebp
  ret

add:
  pushl %ebp
  movl %esp, %ebp
  subl %16, %esp
  mov %3, -4(%ebp)
  movl -4(%esp), %eax
  leave
  ret
```

在真实的物理机器上，执行函数调用是主要包含以下几个步骤:
- 保存调用者栈基地址(%ebp)，当前IP寄存器入栈（即调用者中的下一条指令地址入栈）；
- 调用函数时，在X86平台上，参数从右到左依次入栈；
  - 被调用函数也是从右到左的获取调用参数；
- 一个方法所分配的栈空间大小，取决于该方法内部的局部变量空间、为被调用者所传递的入参大小；
- 被调用者在接收入参时，从`8(%ebp)`处开始，往上逐个获取每一个入参参数；
  - 32位机器，eip会自动入栈，ebp被强制入栈，共8个字节；上述所说的ebp是新的基地址，被调用函数的；
- 被调用者将返回结果保存在`eax`寄存器中，调用者从该寄存器中获取返回值。

Java字节吗指令直接对应一段特定逻辑的本地机器码，而JVM在解释执行Java字节码指令时，会直接调用字节码指令所对应的本地机器吗。这种技术实现的关键便是使用C语言所提供的一种高级功能---函数指针，通过函数指针能够直接由C程序触发一段机器指令。
在JVM内部，call_stub便是实现C程序调用字节码指令的第一步，在JVM执行JAVA主函数对应的第一条字节码指令之前，必须经过call_stub函数指针进入对应的例程，然后再目标例程中触发对Java主函数第一条字节码指令的调用。

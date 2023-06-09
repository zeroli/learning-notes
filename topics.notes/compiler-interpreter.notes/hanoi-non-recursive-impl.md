# hanoi程序的非递归代码实现

递归代码执行其实就是一些程序状态的转移，或者说是状态机

```c++
#if 0
void hanoi(int n, char from, char to, char via)
{
    if (n == 1) {  // 指令0 (pc=0)
        printf("%c => %c\n", from, to);
    } else {
        hanoi(n - 1, from, via, to);  // 指令1 (pc=1)
        hanoi(1,      from, to, via);   // 指令2 (pc=2)
        hanoi(n - 1, via, to, from);  // 指令3 (pc=3)
    }
    // ret // 指令4 (pc=4)
}
#endif

// 每个栈帧保存着被调用函数的入口指令地址（或者是偏移），和入参
struct Frame {
    int pc, n;
    char from, to, via;
};

// 调用一个函数就是将参数入栈，同时将pc设置为函数执行的第一条指令
#define call(...) ({ \
    *(++top) = Frame{.pc = 0, __VA_ARGS__}; \
})

// 函数返回就是将栈帧弹出
#define ret() (--top)
// 无条件跳转指令就是将当前执行指令设置为某一个位置
#define jmp(loc) (top->pc = (loc) - 1)

void hanoi(int n, char from, char to, char via)
{
    Frame stk[64], *top = stk - 1;
    // 调用函数：建立参数调用栈，入参设置为n，从函数第一条指令开始执行
    call(n, from, to, via);
    // 顺序执行指令，pc不断递增，直到调用栈里空了
    for (Frame* f = top; (f = top) >= stk; f->pc++) {
        // 函数开始执行，将入参取出放入局部变量
        n = f->n; from = f->from; to = f->to; via = f->via;
        // 顺序执行，函数指令地址，简单命名为0，1，2，3，4
        switch (f->pc) {
            case 0: {
                if (n == 1) {
                    printf("%c => %c\n", from, to);
                    jmp(4);  // 跳转到指令4，函数返回指令
                    // 并没有实际跳转，只是设置PC到那条指令，接下来执行的就是那条指令
                }
                break;
            }
            case 1: {
                // 递归调用函数，设置函数调用栈，入参和跳转pc（pc=0)
                // 注意，这里仅仅是程序状态的转移，指令的跳转
                // 注意，这里将pc设置为了0，所以接下来就是重复调用函数，从头开始，只是入参不同了而已。
                // 当这次调用序列结束时，程序将会从case 4返回，然后回到当前调用序列，指令递增，从而执行case 2
                call(n-1, from, via, to);
                break;
            }
            case 2: {
                call(1, from, to, via);
                break;
            }
            case 3: {
                call(n-1, via, to, from);
                break;
            }
            case 4: {
                ret();
                break;
            }
            default:
                assert(0);
        }
    }
}

int main()
{
    hanoi(3, 'A', 'B', 'C');
}
```
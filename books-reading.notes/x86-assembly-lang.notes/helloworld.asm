section .data  ; 数据段
    msg db "Hello world!", 0

section .text  ; 代码段
    global _start

_start:
    mov rax, 1      ; rax，系统调用号，1
    mov rdi, 1      ; 第一次参数放在rdi中
    mov rsi, msg    ; 第二个参数放在rsi中
    mov rdx, 13     ; 第三个参数放在rdx中,     
    syscall         ; 启动系统调用
    mov rax, 60     ; 系统调用号: 60
    mov rdi, 0      ; 第一次参数放在rdi中
    syscall         ; 启动系统调用: exit(0)
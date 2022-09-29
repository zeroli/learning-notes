- 信号的发送与处理是一个复杂的过程，这里来总结一下。
  1. 假设我们有一个进程 A，main 函数里面调用系统调用进入内核。
  2. 按照系统调用的原理，会将用户态栈的信息保存在 pt_regs 里面，也即记住原来用户态是运行到了 line A 的地方。
  3. 在内核中执行系统调用读取数据。
  4. 当发现没有什么数据可读取的时候，只好进入睡眠状态，并且调用 schedule 让出 CPU，这是进程调度第一定律。
  5. 将进程状态设置为 TASK_INTERRUPTIBLE，可中断的睡眠状态，也即如果有信号来的话，是可以唤醒它的。
  6. 其他的进程或者 shell 发送一个信号，有四个函数可以调用 kill、tkill、tgkill、rt_sigqueueinfo。
  7. 四个发送信号的函数，在内核中最终都是调用 do_send_sig_info。
  8. do_send_sig_info 调用 send_signal 给进程 A 发送一个信号，其实就是找到进程 A 的 task_struct，或者加入信号集合，为不可靠信号，或者加入信号链表，为可靠信号。
  9. do_send_sig_info 调用 signal_wake_up 唤醒进程 A。
  10. 进程 A 重新进入运行状态 TASK_RUNNING，根据进程调度第一定律，一定会接着 schedule 运行。
  11. 进程 A 被唤醒后，检查是否有信号到来，如果没有，重新循环到一开始，尝试再次读取数据，如果还是没有数据，再次进入 TASK_INTERRUPTIBLE，即可中断的睡眠状态。
  12. 当发现有信号到来的时候，就返回当前正在执行的系统调用，并返回一个错误表示系统调用被中断了。
  13. 系统调用返回的时候，会调用 exit_to_usermode_loop。这是一个处理信号的时机。
  14. 调用 do_signal 开始处理信号。
  15. 根据信号，得到信号处理函数 sa_handler，然后修改 pt_regs 中的用户态栈的信息，让 pt_regs 指向 sa_handler。同时修改用户态的栈，插入一个栈帧 sa_restorer，里面保存了原来的指向 line A 的 pt_regs，并且设置让 sa_handler 运行完毕后，跳到 sa_restorer 运行。
  16. 返回用户态，由于 pt_regs 已经设置为 sa_handler，则返回用户态执行 sa_handler。
  17. sa_handler 执行完毕后，信号处理函数就执行完了，接着根据第 15 步对于用户态栈帧的修改，会跳到 sa_restorer 运行。
  18. sa_restorer 会调用系统调用 rt_sigreturn 再次进入内核。
  19. 在内核中，rt_sigreturn 恢复原来的 pt_regs，重新指向 line A。
  20. 从 rt_sigreturn 返回用户态，还是调用 exit_to_usermode_loop。
  21. 这次因为 pt_regs 已经指向 line A 了，于是就到了进程 A 中，接着系统调用之后运行，当然这个系统调用返回的是它被中断了，没有执行完的错误。
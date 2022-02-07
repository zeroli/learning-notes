# protobuf-2.6.1源码阅读

## subprocess的代码阅读
file: src\google\protobuf\compiler\subprocess.h
file: src\google\protobuf\compiler\subprocess.cc

`SubProcess`fork一个子进程，运行另一个程序，然后跟它进行通信，等待结果的实现。
划重点：
* 通信采用的PIPE，serialize/deserialize采用的是简易的Protobuf's Message。
* 输入输出等待采用`Select`多播的方式；
* 执行子进程程序，有不同的方式：
```c++
  enum SearchMode {
    SEARCH_PATH,   // Use PATH environment variable.
    EXACT_NAME     // Program is an exact file name; don't use the PATH.
  };

    switch (search_mode) {
      case SEARCH_PATH:
        execvp(argv[0], argv);
        break;
      case EXACT_NAME:
        execv(argv[0], argv);
        break;
    }
```
> 注意这里`execvp`和`execv` 的使用。

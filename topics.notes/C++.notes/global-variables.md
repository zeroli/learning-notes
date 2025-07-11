## Introduction
--------------------
This page will list some cases which are involved with global variable uses in C/C++, about their tricks and pitfalls, to better utilize global variables in C++ programming.

增加一个case来说明函数内的static 指针变量直接new出来,
valgrind不会报memory leak

## Cases
simplest case:
- define global variable primitive type(int) and inititialized from lambda expression in binary
```c++
#include <iostream>

int g_x = []() -> int {
    std::cerr << "init global int x before main\n";
    return 10;
}();

int main()
{
    std::cerr << "\n==main==\n";
    std::cerr << "global x = " << g_x << "\n";
}
```
- build command
`g++ main.cc -o main && ./main`
- output:
```sh
init global int x before main

==main==
global x = 10
```
- comment
even though the global variable is a primitive type int, no ctor for it, there is init code defined by lambda expression that needs to be run before main.


- define global variable whose ctor is not trivial in binary
```c++
     1	#include <iostream>
     2
     3	struct MyStruct {
     4	    MyStruct() {
     5	        std::cerr << "ctor for MyStruct\n";
     6	    }
     7	    ~MyStruct() {
     8	        std::cerr << "dtor for MyStruct\n";
     9	    }
    10
    11	    int x = 10;
    12	};
    13
    14	MyStruct mystruct;
    15
    16	int main()
    17	{
    18	    std::cerr << "\n==main==\n";
    19	    std::cerr << "global x = " << mystruct.x << "\n";
    20	}
    21
```
- output
```sh
ctor for MyStruct

==main==
global x = 10
dtor for MyStruct
```
- comment
once MyStruct has ctor, whetherever it's defined by us or synthesized by compiler, ctor will be invoked before main, and dtor will be invoked after main

- define global variable whose ctor is not trivial in static lib
  - mystruct.cc (mystruct.h same as previous)
```c++
 2	#include "mystruct.h"
 3
 4	MyStruct mystruct;
```
  - main.cc
```c++
 1	#include <iostream>
 2	#include "mystruct.h"
 3
 4	extern MyStruct mystruct;
 5
 6	int main()
 7	{
 8	    std::cerr << "\n==main==\n";
 9	    std::cerr << "global mystruct = " << mystruct.x << "\n";
10	}
```

- build command:
```sh
g++ -c mystruct.cc -o mystruct.o && ar sr libmystruct.a mystruct.o
g++ main.cc -L. -lmystruct -o main
```

- output
same as previous output (global object variable defined and built into binary)

- comment
  - binary code refer that global object variable through `extern`
  - otherwise, use flag `-Wl,--whole-archive -lmystruct -Wl,--no-whole-archive` to tell gcc to include all of libmystruct.a


## global variables in different cpp files(different compilation units), different static libs
    - Global variable init order, dependency, uninit order (dependency), app atExit??
    - Global variables in static lib, how to link into final binary?
    - Global variables in dynlib, init/uninit order/time? dlopen/dlclose? Dlclose will destroy global variables?
    - Dlopen => dlclose => dlopen, global variable re-init? TF?



##
- binary calls a function which defines static variable, from a static lib
- a shared library loaded by binary also calls that function from that static lib, and link that static lib to shared lib

- main.cc
```c++
     1	#include <iostream>
     2	#include "mystruct.h"
     3	#include <dlfcn.h>
     4	#include <cassert>
     5
     6	int main()
     7	{
     8	    std::cerr << "\n==main==\n";
     9	    std::cerr << "calling global_f(): " << (void*)&global_f << " from static lib\n"
    10	              << global_f().x << "\n";
    11
    12	    std::cerr << "\nloading dync lib\n";
    13	    void* f = dlopen("./libdynlib.so", RTLD_NOW);
    14	    assert(f);
    15	}
    16
```
  - `g++ main.cc -o main -L. -lmystruct -ldl && ./main`
- mystruct.cc
```c++
     1	#include <iostream>
     2	#include "mystruct.h"
     3
     4	MyStruct& global_f()
     5	{
     6	    std::cerr << "\nwe're in static mystruct lib\n"
     7	              << "global mystruct from function static variable\n";
     8	    static MyStruct mystruct;
     9	    std::cerr << "&mystruct=" << (void*)&mystruct << "\n";
    10	    return mystruct;
    11	}
    12
```
  - `g++ -fPIC -c mystruct.cc -o mystruct.o && ar sr libmystruct.a mystruct.o`
- dynlib.cc
```c++
     1	#include "mystruct.h"
     2	#include <iostream>
     3
     4	int x = [] {
     5	    std::cerr << "\nwe're running in dynlib\n"
     6	              << "calling global_f(): " << (void*)&global_f << "\n";
     7	    auto& mystruct = global_f();
     8	    return mystruct.x;
     9	}();
    10
```
  - `g++ -fPIC -shared dynlib.cc -L. -lmystruct -o libdynlib.so`

- output
```sh

==main==
calling global_f(): 0x400b2a from static lib  /// *** global_f address 1

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x6021bc
10

loading dync lib

we're running in dynlib
calling global_f(): 0x7fb74d129c45  /// *** global_f address 2

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x7fb74d32b094
dtor for MyStruct
dtor for MyStruct
```
- comment
  - both binary and shared library contain the function `global_f` code, they call their own function determined by linker at build time
  - note: build command for binary main doesn't have `-rdynamic`, so that the function/symbol `global_f` is not placed to dynamic section, and loaded shared library cannot lookup/resolve it and lookup its own `global_f`

## add `-rdynamic` flag to build command for binary main
- `g++ main.cc -o main -L. -lmystruct -ldl -rdynamic && ./main`
- output
```sh

==main==
calling global_f(): 0x400dda from static lib  /// *** global_f address

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x6021bc
10

loading dync lib

we're running in dynlib
calling global_f(): 0x400dda  /// *** global_f address

we're in static mystruct lib
global mystruct from function static variable
&mystruct=0x6021bc
dtor for MyStruct
```
- comment
  - same function `global_f` address is referenced, and that global function-scope static object `mystruct`


## add `-rdynamic` flag to build command for binary main, but hide symbol `global_f` in binary main
- `g++ main.cc -o main -L. -lmystruct -ldl -rdynamic -Xlinker --version-script=ld.version && ./main`
  - adding `-fvisibility=hidden` not working
  - ld.version file: `{ local: *global_f*; };`

- output
```sh

==main==
calling global_f(): 0x400d7a from static lib  /// *** global_f address 1

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x6021bc
10

loading dync lib

we're running in dynlib
calling global_f(): 0x7f720d083c45  /// *** global_f address 2

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x7f720d285094
dtor for MyStruct
dtor for MyStruct
```
- comment
  - once the symbol `global_f` in binary main is hidden, runtime linker cannot lookup but have to find one for shared lib

## shared lib build doesn't include -lmystruct static lib in its link flag
- `g++ -fPIC -shared dynlib.cc -o libdynlib.so` (no `-L. -lmystruct`)
- shared library will not contain the init ctor code for global variable
- shared library refering symbols from static lib will be left unresolved, expecting resolved at runtime
```sh
$nm -Cu libdynlib.so | grep global_f
                 U global_f()
```
- run binary with hidden symbols, and load this shared library with unresolved symbols:
```sh
==main==
calling global_f(): 0x400d7a from static lib

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x6021bc
10

loading dync lib
main: main.cc:14: int main(): Assertion `f' failed.   /// *** load failed, init code failed to access unresolved global_f symbol
Abort (core dumped)
```
- run binary with exposed symbols, and load this shared library with unresolved symbols:
```sh
==main==
calling global_f(): 0x400dda from static lib  /// *** global_f address

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x6021bc
10

loading dync lib

we're running in dynlib
calling global_f(): 0x400dda  /// *** global_f address

we're in static mystruct lib
global mystruct from function static variable
&mystruct=0x6021bc
dtor for MyStruct
```
  - now, shared lib refers to symbol global_f from binary main

## binary main hides symbol, load 2 shared libs, both refers to same symbols from static libs
- output: there are 3 symbols to access, 3 global variables total
```sh
==main==
calling global_f(): 0x400dbc from static lib  // *** global_f address 1

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x6021bc
10

loading dync lib1

we're running in dynlib
calling global_f(): 0x7f7ddd1bbc45  // *** global_f address 2

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x7f7ddd3bd094

loading dync lib2

we're running in dynlib
calling global_f(): 0x7f7ddcfb8c45  // *** global_f address 3

we're in static mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x7f7ddd1ba094
dtor for MyStruct
dtor for MyStruct
dtor for MyStruct
```
- if append `RTLD_GLOBAL` for shared library in binary main, the loaded symbols from shared lib will be seen by any subsequent lib loading. For this case, if binary main hides symbols, then 2 shared libaries will have same symbol to refer.

## `global_f` is defined in shared lib, statically loaded by binary main.
- `g++ -fPIC -shared mystruct.cc -o libmystruct.so`
- `g++ main.cc -o main ./libmystruct.so -ldl -rdynamic && ./main`
- output:
```sh
==main==
calling global_f(): 0x400780 from lib  // *** global_f address

we're in mystruct lib
global mystruct from function static variable
ctor for MyStruct
&mystruct=0x7fecb759e084
10

loading dync lib1

we're running in dynlib
calling global_f(): 0x400780 // *** global_f address

we're in mystruct lib
global mystruct from function static variable
&mystruct=0x7fecb759e084
```
  - binary main => static load libmystruct.so
  - binary main => dynamic load libdynlib.so
  - dlopen-ed shared library will refer to same symbol `global_f` as binary main, exported by libmystruct.so

```c++
// foo.cc
int x1;
int x2 = 10;
std::string x3;
std::string x4 = "hello";
static int x5{10};
static std::string x6{"hello"};
void foo() {
  static int x1;
  static std::string x2;
}
struct Foo {
  static const int x1 = 10;                // c++98
  static float x1f;                        // c++98
  static std::string x4;                   // c++98

  static constexpr int x2 = 20;            // c++11
  static inline float x3 = 30.f;           // c++17
  static inline std::string x5 = "hello";  // c++17
};
float Foo::x1f = 10.f;
std::string Foo::x4{"hello"};

namespace NS {
  static int x1;
  int x2;
  std::string x3("hello");
}  // namespace NS

extern int X;  // declaration of external global `X`

```

```c++
#include "MyStruct1.h"
MyStruct1 global_mystruct1;
void main() {
}
```

```c++
// MyStruct1.h
#include "MyStruct2.h"

extern MyStruct2 global_mystruct2;

struct MyStruct1 {
  MyStruct1() {
    global_mystruct2.log();
  }
  ~MyStruct1() {
    global_mystruct2.log();
  }
};

// MyStruct2.h
struct MyStruct2 {
  //...
  void log() {
    // ...
  }
};
MyStruct2 global_mystruct2;

MyStruct2& GetMyStruct2();

// MyStruct2.cc
MyStruct2& GetMyStruct2() {
  static MyStruct2 static_struct2;
  return static_struct2;
}

int global_var = foo();

int& foo()
{
  static int s_int = [] {
    /// heavy operations here...
  }();
  return s_int;
}

static std::atomic<bool> g_init{false};
static std::mutex g_mtx;
int& foo()
{
  static int s_int;
  if (!g_int) {
    std::lock_guard<std::mutex> lock(g_mtx);
    s_int = [] { ... }();
    g_init = true;
  }
  return s_int;
}

Singleton& GetInstance()
{
  static Singleton singleton;
  return singleton;
}

Singleton& GetInstance()
{
  static Singleton* psingleton = new Singleton();
  return *psingleton;
}
```

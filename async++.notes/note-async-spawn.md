# async::spawn学习

Example代码
=======
```CPP {.numberLines}
#include <async++.h>
auto task1 = async::spawn([] {
    std::cout << "Task 1 executes asynchronously" << std::endl;
});
```
- 要实现上面代码的功能，后台至少应该有一个线程池在运行，然后将这个async task schedule到其中一个线程运行。返回的`task1`应该是库提供的一个类的对象，比如说是`Task`类；
- `spawn`函数应该为一个utility函数模板，接收一个callable object。比如说`Func&& func`；

下面来深入进去看看它是怎么被实现出来的：
```CPP {.numberLines, .file=include\async++\task.h}
template<typename Func>
decltype(async::spawn(::async::default_scheduler(), std::declval<Func>())) spawn(Func&& f)
{
	return async::spawn(::async::default_scheduler(), std::forward<Func>(f));
}
```
- 实现的方法跟我们猜测的类似，函数模板，接收`Func&& f`参数；
- 这里用了`std::declval<Func>()`技术，它并没有真正构造一个`Func`对象，用在了`decltype`语境中用来推导函数的返回值类型；
- 这个utility函数调用了另一个overloaded函数，接收2个参数，第一个参数为默认的临时构造的scheduler类型对象；

我们来看看另一个具体干活的重载函数：
```CPP {.numberLines}
template<typename Sched, typename Func>
task<
  typename detail::remove_task<
    typename std::result_of<
      typename std::decay<Func>::type()
    >::type
  >::type
>
spawn(Sched& sched, Func&& f) { ... }
```
- 函数接收模板参数`Sched`和`Func`
- 返回一个叫做`task`的模板类对象
    - `typename std::decay<Func>::type`：`Func`decay之后的类型，比如传入函数名字，decay成函数指针
    - `typename std::result_of<typename std::decay<Func>::type()>::type`：构造一个decay之后的`Func`对象，调用默认构造函数，应用`std::result_of`获取它的类型
    - `typename detail::remove_task(...)::type`：应用detail::remove_task进行traits操作（TODO）
    - 用上面traits操作之后的类型作为`task`类的模板实例化参数

问题？
- 为啥不用std::declval和decltype的组合来获取一个Func decay之后的具体类型?
> 函数实现开头代码中有这样的注释：
> 	// Using result_of in the function return type to work around bugs in the Intel
	// C++ compiler.

- 首先确保传入的`Func`是可被调用的。**(TODO：这个我们可以学习并应用下）**
```CPP {.numberLines}
// Make sure the function type is callable
typedef typename std::decay<Func>::type decay_func;
static_assert(detail::is_callable<decay_func()>::value, "Invalid function type passed to spawn()");
```
- 接着创建task对象
```CPP {.numberLines}
// Create task
typedef typename detail::void_to_fake_void<typename detail::remove_task<decltype(std::declval<decay_func>()())>::type>::type internal_result;
typedef detail::root_exec_func<Sched, internal_result, decay_func, detail::is_task<decltype(std::declval<decay_func>()())>::value> exec_func;
task<typename detail::remove_task<decltype(std::declval<decay_func>()())>::type> out;
detail::set_internal_task(out, detail::task_ptr(new detail::task_func<Sched, exec_func, internal_result>(std::forward<Func>(f))));
```
上面第4行创建一个`task`对象`out`，采用的就是std::declval和decltype的组合技术
    - `std::declval<decay_func>()()`：decay_func的具体类型，然后"调用"构造函数（其实并不会）；
    - 采用`decltype`推导类型，注意decltype类似于sizeof，是一个操作符，使用方式类似于函数调用。它不是函数或类模板；
最后调用`set_internal_task`，好像是将task对象`out`与`f`进行绑定。**（TODO？？）**

- 接着就将包装的task，shedule到一个线程中运行
```CPP {.numberLines}
	// Avoid an expensive ref-count modification since the task isn't shared yet
	detail::get_internal_task(out)->add_ref_unlocked();
	detail::schedule_task(sched, detail::task_ptr(detail::get_internal_task(out)));
```

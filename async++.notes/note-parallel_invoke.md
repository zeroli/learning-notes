# parallel_invoke的实现学习

`parallel_invoke`
=====

example code：
```CPP {.numberLines}
    async::parallel_invoke([] {
        std::cout << "This is executed in parallel..." << std::endl;
    }, [] {
        std::cout << "with this" << std::endl;
    });
```
> 提供N个 tasks并发提交到default_scheduler执行（default_scheduler就是线程池scheduler)
```CPP {.numberLines}
template<typename... Args>
void parallel_invoke(Args&&... args)
{
	async::parallel_invoke(::async::default_scheduler(), std::forward<Args>(args)...);
}
```
```CPP {.numberLines}
template<typename Sched, typename... Args>
typename std::enable_if<detail::is_scheduler<Sched>::value>::type parallel_invoke(Sched& sched, Args&&... args)
{
	detail::parallel_invoke_internal<0, sizeof...(Args)>::run(sched, std::forward_as_tuple(std::forward<Args>(args)...));
}
```
- enable_if的使用，只有模板参数`Sched`是一个真正的scheduler时，这个函数才会调用；
- 函数体实现是将输入的用户代码函数包装成tuple（`forward_as_tuple`)，然后sched一个个的按顺序放到线程池的队列里；
- 因为支持任意的用户代码函数，故代码实现其实是想iterate tuple的元素，但是C++11并不直接支持tuple iteration，因此代码实现是用index 编译期迭代的方式；
```CPP {.numberLines}
template<std::size_t Start, std::size_t Count>
struct parallel_invoke_internal {
	template<typename Sched, typename Tuple>
	static void run(Sched& sched, const Tuple& args)
	{
		auto&& t = async::local_spawn(sched, [&sched, &args] {
			parallel_invoke_internal<Start + Count / 2, Count - Count / 2>::run(sched, args);
		});
		parallel_invoke_internal<Start, Count / 2>::run(sched, args);
		t.get();
	}
};
```
主模板类的模板参数是`Start`和`Count`，tuple element 的起始index和个数；

提供2个特化类实现，终止递归：
```CPP {.numberLines}
template<std::size_t Index>
struct parallel_invoke_internal<Index, 1> {
	template<typename Sched, typename Tuple>
	static void run(Sched&, const Tuple& args)
	{
		// Make sure to preserve the rvalue/lvalue-ness of the original parameter
		std::forward<typename std::tuple_element<Index, Tuple>::type>(std::get<Index>(args))();
	}
};
template<std::size_t Index>
struct parallel_invoke_internal<Index, 0> {
	template<typename Sched, typename Tuple>
	static void run(Sched&, const Tuple&) {}
};
```
- 当递归到某一个element时，它的`run`如下：
  - 调用`tuple_element<Index, Tuple>`获得对应element的类型；
  - 调用`get<Index>(args)`获得对应element的值引用；
  - 然后`std::forward`，注释说要保持原有参数的左值或右值特性。但是为啥？这里直接触发了？？？==**TODO**==
  - 最后调用`operator ()`
- 当递归到没有element时，它的`run`是空的；

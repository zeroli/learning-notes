# parallel_for的实现学习

example
=======
```CPP {.numberLines}
async::parallel_for(async::irange(0, 5), [](int x) {
    std::cout << x;
});
```
`irange`实现了python中的irange的功能: 提供一对begin/end，返回一个可以进行iterate的对象；
```CPP {.numberLines}
template<typename T, typename U>
int_range<typename std::common_type<T, U>::type> irange(T begin, U end)
{
	return {begin, end};
}
```
- 这里简单的实现了int_range对象，从begin/end来构造；
- `int_range`类必然会提供`begin()`和`end()`接口，以便于迭代操作；
    ```CPP {.numberLines}
    iterator begin() const
    {
        return iterator(value_begin);
    }
    iterator end() const
    {
        return iterator(value_end);
    }
    ```
- 从`value_begin`和`value_end`直接构造出`iterator`对象；
- int_range的`iterator`类满足C++关于iterator的一些条件，譬如内嵌的typedef:
   ```CPP {.numberLines}
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef iterator pointer;
    typedef T reference;
    typedef std::random_access_iterator_tag iterator_category;
    ```
- 定义了前向和后向的operator ++/--操作；


`parallel_for`的实现
======
```CPP {.numberLines}
template<typename Range, typename Func>
void parallel_for(Range&& range, const Func& func)
{
	async::parallel_for(::async::default_scheduler(), range, func);
}
```
同时它还可以这样的调用: 直接提供一个initializer_list进行构造迭代：
```CPP {.numberLines}
template<typename T, typename Func>
void parallel_for(std::initializer_list<T> range, const Func& func)
{
	async::parallel_for(async::make_range(range.begin(), range.end()), func);
}
```
`make_range`utility函数会将一对iterator封装成一个简单的类，从而适应`Range`的调用，这个简单的类也需要提供begin/end接口；
```CPP {.numberLines}
template<typename Iter>
class range {
	Iter iter_begin, iter_end;

public:
	range() = default;
	range(Iter a, Iter b)
		: iter_begin(a), iter_end(b) {}

	Iter begin() const
	{
		return iter_begin;
	}
	Iter end() const
	{
		return iter_end;
	}
};

// Construct a range from 2 iterators
template<typename Iter>
range<Iter> make_range(Iter begin, Iter end)
{
	return {begin, end};
}
```

```CPP {.numberLines}
template<typename Sched, typename Range, typename Func>
void parallel_for(Sched& sched, Range&& range, const Func& func)
{
	detail::internal_parallel_for(sched, async::to_partitioner(std::forward<Range>(range)), func);
}
```
调用将`to_partitioner`将`Range`转换成`Partitioner`，调用内部实现函数；
可以猜想作者应该会采用跟`parallel_invoke`类似的算法，每次将range进行折半，封装成task，schedule到线程池里运行，每个task做同样的事情，折半，封装成task，schedule到线程池运行，直到task的range只有一个元素要处理。如此一来每个func都会在一个单独的线程中启动并运行，从而实现parallellism；
```CPP {.numberLines}
template<typename Sched, typename Partitioner, typename Func>
void internal_parallel_for(Sched& sched, Partitioner partitioner, const Func& func)
{
	// Split the partition, run inline if no more splits are possible
	auto subpart = partitioner.split();
	if (subpart.begin() == subpart.end()) {
		for (auto&& i: partitioner)
			func(std::forward<decltype(i)>(i));
		return;
	}

	// Run the function over each half in parallel
	auto&& t = async::local_spawn(sched, [&sched, &subpart, &func] {
		detail::internal_parallel_for(sched, std::move(subpart), func);
	});
	detail::internal_parallel_for(sched, std::move(partitioner), func);
	t.get();
}
```
`local_spawn`会将封装的lambda函数包装成task丢到线程池中运行，这个task会处理后一半的range;
前一半的range继续在当前线程中递归处理；


partitioner
=========
```
// Partitioners are essentially ranges with an extra split() function. The
// split() function returns a partitioner containing a range to be executed in a
// child task and modifies the parent partitioner's range to represent the rest
// of the original range. If the range cannot be split any more then split()
// should return an empty range.
```

如何判断一个类是partitioner？
======
```CPP {.numberLines}
// Detect whether a range is a partitioner
template<typename T, typename = decltype(std::declval<T>().split())>
two& is_partitioner_helper(int);
template<typename T>
one& is_partitioner_helper(...);
template<typename T>
struct is_partitioner: public std::integral_constant<bool, sizeof(is_partitioner_helper<T>(0)) - 1> {};
```
如果一个类有`split`成员函数，则它可以判断为`partitioner`类；
==> 这个判断方法有点牵强
```CPP {.numberLines}
	static_partitioner_impl split()
	{
		// Don't split if below grain size
		std::size_t length = std::distance(iter_begin, iter_end);
		static_partitioner_impl out(iter_end, iter_end, grain);
		if (length <= grain)
			return out;

		// Split our range in half
		iter_end = iter_begin;
		std::advance(iter_end, (length + 1) / 2);
		out.iter_begin = iter_end;
		return out;
	}
```
`auto_partitioner_impl`类的split函数实现有点复杂；

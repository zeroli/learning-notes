# parallel_reduce的实现学习

parallel_reduce
=======
example代码：
```CPP {.numberLines}
    int r = async::parallel_reduce({1, 2, 3, 4}, 0, [](int x, int y) {
        return x + y;
    });
```
先对每个元素进行map操作，然后对map结果进行reduce操作；
基本的框架代码与`parallel_for`或者`paralle_invoke`类似，核心代码如下：
```CPP {.numberLines}
template<typename Sched, typename Partitioner, typename Result, typename MapFunc, typename ReduceFunc>
Result internal_parallel_map_reduce(Sched& sched, Partitioner partitioner, Result init, const MapFunc& map, const ReduceFunc& reduce)
{
	// Split the partition, run inline if no more splits are possible
	auto subpart = partitioner.split();
	if (subpart.begin() == subpart.end()) {
		Result out = init;
		for (auto&& i: partitioner)
			out = reduce(std::move(out), map(std::forward<decltype(i)>(i)));
		return out;
	}

	// Run the function over each half in parallel
	auto&& t = async::local_spawn(sched, [&sched, &subpart, init, &map, &reduce] {
		return detail::internal_parallel_map_reduce(sched, std::move(subpart), init, map, reduce);
	});
	Result out = detail::internal_parallel_map_reduce(sched, std::move(partitioner), init, map, reduce);
	return reduce(std::move(out), t.get());
}
```

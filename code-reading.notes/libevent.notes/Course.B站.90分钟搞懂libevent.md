# Course.B站.90分钟搞懂libevent

## 服务端事件构成
* 网络事件
* 定时事件
* 信号事件

Reactor模型
====
The reactor design patter is an event handling pattern for handling service requests delivered concurrently to a service handler by one or more inputs. The service pattern then demultiplexes the incoming requests and dispatches them synchronously to the associated request handlers.

* 事件驱动
  将网络IO处理转化为事件处理

* 处理一个或多个并发传递到服务端的服务请求
> IO多路复用：
> 不同平台有不同实现：
> 1. mac: kqueue
> 2. linux: select, poll以及epoll
> 3. windows: IOCP
多路：同时监控多个IO事件，非阻塞
复用：用一个线程来处理多个事件（减少线程）

* 对传入的请求进行解复用并**同步**分派到关联handler
  事件循环：事件检测以及事件分派
  同步：一个事件一个事件进行处理

libevent是针对reactor的封装，跨平台
event: 事件
event_base: 事件管理器

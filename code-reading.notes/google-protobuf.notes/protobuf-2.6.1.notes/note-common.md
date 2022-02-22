# protobuf-2.6.1源码阅读

## common工具类的代码阅读
### `scoped_ptr`和`scoped_array_ptr`的实现


### `GOOGLE_COMPILE_ASSERT`
```c++
#define GOOGLE_COMPILE_ASSERT(expr, msg) \
  typedef ::google::protobuf::internal::CompileAssert<(bool(expr))> \
          msg[bool(expr) ? 1 : -1]
```
* `#define COMPILE_ASSERT(expr, msg) typedef char msg[(expr) ? 1 : -1]`，这种方式不会work，因为`expr`是在运行期评估的；
* `(bool(expr))`加上外围的`()`，是work around这样的代码: `5 > 1`，因为`>`会被误解为模板参数列表终止符；
* `bool(expr) ? 1 : -1`，将`expr`转换为`bool`，是为了这样的代码`((0.0) ? 1 : -1)`可以工作，MSVC有bug

### Google LOG
这里我们又看到了是如何打log的了：
```c++
#define GOOGLE_LOG(LEVEL)                                                 \
  ::google::protobuf::internal::LogFinisher() =                           \
    ::google::protobuf::internal::LogMessage(                             \
      ::google::protobuf::LOGLEVEL_##LEVEL, __FILE__, __LINE__)
#define GOOGLE_LOG_IF(LEVEL, CONDITION) \
  !(CONDITION) ? (void)0 : GOOGLE_LOG(LEVEL)
```
比如：`GOOGLE_LOG(INFO) << "this is one message";`
`LogMessage`类提供operator <<操作符，然后`LogFinisher`类的operator=接收一个`LogMessage`，然后调用LogMessage的一个成员函数将所有的message输出到目标对象中。
```c++
void LogFinisher::operator=(LogMessage& other) {
  other.Finish();
}
```

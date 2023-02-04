下面节选自Tensorflow的代码： 用macro来注册operation：
```c++
// REGISTER_OP("my_op_name")
//     .Attr("<name>:<type>")
//     .Attr("<name>:<type>=<default>")
//     .Input("<name>:<type-expr>")
//     .Input("<name>:Ref(<type-expr>)")
//     .Output("<name>:<type-expr>")
//     .Doc(R"(comment)");
```
宏REGISTER_OP接受一个字符串表示操作的名字，展开成一个东西，应该后面可以直接跟着某个对象的方法函数.Attr/.Input/.Output
只要词法上符合语法就可以了，比如展开成这样：
`static const A a = B(name)`
`B(name)`构造一个B的临时对象，可以直接调用它的方法函数.Attr(...), .Input(...)，或者.Output(...)
`static const A a = B(name).Attr(...).Input(...).Output(...);`

构造A类的对象a，应该是非常轻量级的。而且要从B来构造A，那么A的唯一的构造函数就应该是A(const B)。
然后在A的构造函数中，对B的那个临时对象进行处理，比如继续调用B的build函数，生成一个最终结果。

来看下TF的代码：
```c++
#define REGISTER_OP_UNIQ(ctr, name)                                          \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr    \
      TF_ATTRIBUTE_UNUSED =                                                  \
      // 这里B其实是一个模板函数，针对某些不进行构造，那个模板特化类是空类，可以被编译器优化掉
          ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP( \  
              name)>(name)
 
namespace register_op {
OpDefBuilderReceiver::OpDefBuilderReceiver(  // A的构造函数，接受const B&
    const OpDefBuilderWrapper<true>& wrapper) {
  OpRegistry::Global()->Register(
      [wrapper](OpRegistrationData* op_reg_data) -> Status {
        return wrapper.builder().Finalize(op_reg_data);  // 在另一个地方调用B的某个方法函数
      });
}
```


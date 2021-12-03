# identity_op kernel的学习

tensorflow\core\kernels\identity_op.h
这个kernel应该是最简单的那种kernel，将输入直接identify为输出。

```CPP
class IdentityOp : public OpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};
```

这个`IdentityOp`继承于公共基类`OpKernel`，实现接口函数`Compute`，是一个同步kernel。

注意：基类`OpKernel`是定义在: tensorflow\core\framework\op_kernel.h，framework目录里面

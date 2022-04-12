```c++
template <newronplus::DataTypeEnum>
struct NpToTfDataTypeEnum;
template <>
struct NpToTfDataTypeEnum<newronplus::DataTypeEnum::INT32> {
    static constexpr tensorflow::DataType value = tensorflow::DT_INT32;
};
template <>
struct NpToTfDataTypeEnum<newronplus::DataTypeEnum::FLOAT> {
    static constexpr tensorflow::DataType value = tensorflow::DT_FLOAT;
};
template <>
struct NpToTfDataTypeEnum<newronplus::DataTypeEnum::DOUBLE> {
    static constexpr tensorflow::DataType value = tensorflow::DT_DOUBLE;
};
template <>
struct NpToTfDataTypeEnum<newronplus::DataTypeEnum::STRING> {
    static constexpr tensorflow::DataType value = tensorflow::DT_STRING;
};
```
当我们有自己的一套enumeration，第三方库也有一套，在代码中经常需要相互转换。
上面的代码利用模板技术定义了一个查找表，从我们自己的enum到第三方库enum的映射。

当然也可以定义另外一组模板特化，从第三方库enum到我们自己的enum的映射。

上面是enum常量的映射，当然也可以定义类型到类型的映射，模板特化类提供嵌套的类型`type`作为目标类型。

# TensorDescInfo

```
struct TensorDescInfo {
    Format format_ = FORMAT_RESERVED;        /* tbe op注册支持的格式 */
    DataType dataType_ = DT_UNDEFINED;       /* tbe op注册支持的数据类型 */
    };
```

Format为枚举类型，定义请参考[Format](./cannkit-ge-format.md)。

DataType为枚举类型，定义请参考[DataType](./cannkit-ge-datatype.md)。  

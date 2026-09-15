# GetDataType

#### 函数功能

获取Tensor的数据类型。  

#### 函数原型

```
ge::DataType GetDataType() const
```

#### 参数说明

无  

#### 返回值

返回Tensor中的数据类型。

关于ge::DataType的定义，请参见[DataType](./cannkit-ge-datatype.md)。  

#### 约束说明

无  

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {}, {}, ge::DT_FLOAT, nullptr};
// ge::DT_FLOAT
auto dt = t.GetDataType();
```


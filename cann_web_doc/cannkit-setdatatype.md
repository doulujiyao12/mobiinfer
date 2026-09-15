# SetDataType

#### 函数功能

设置Tensor的数据类型。  

#### 函数原型

```
void SetDataType(const ge::DataType data_type)
```

#### 参数说明

|参数|输入/输出|说明|
|:--------|:----|:------------------------------------------------------------------------------------------------------------------------------------|
|data_type|输入|需要设置的Tensor的数据类型。 关于ge::DataType的定义，请参见[DataType](./cannkit-ge-datatype.md)。|

#### 返回值

无  

#### 约束说明

无  

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {}, {}, ge::DT_FLOAT, nullptr};
t.SetDataType(ge::DT_DOUBLE);
```


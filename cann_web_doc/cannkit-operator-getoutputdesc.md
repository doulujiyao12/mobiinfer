# GetOutputDesc

#### 函数功能

根据算子Output名称或Output索引获取算子Output的TensorDesc。  

#### 函数原型

![](https://media:401788444114787933)  
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
TensorDesc GetOutputDesc(const std::string &name) const;
TensorDesc GetOutputDescByName(const char_t *name) const;
TensorDesc GetOutputDesc(uint32_t index) const;
```

#### 参数说明

|参数名|输入/输出|描述|
|:----|:----|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|name|输入|算子Output名称。 当无此算子Output名称时，返回TensorDesc默认构造的对象，其中，主要设置[DataType](./cannkit-ge-datatype.md)为DT_FLOAT（表示float类型），[Format](./cannkit-ge-format.md)为FORMAT_NCHW（表示NCHW）。|
|index|输入|算子Output索引。 当无此算子Output索引时，则返回TensorDesc默认构造的对象，其中，主要设置[DataType](./cannkit-ge-datatype.md)为DT_FLOAT（表示float类型），[Format](./cannkit-ge-format.md)为FORMAT_NCHW（表示NCHW）。|

#### 返回值

|类型|描述|
|:-------------------------------------------------------------------------------------------------------------------------|:-------------------|
|[TensorDesc](./cannkit-tensordesc-construction-and-destructor.md)|算子Output的TensorDesc。|

#### 异常处理

无  

#### 约束说明

无  

# 构造函数和析构函数

#### 函数功能

TensorDesc构造函数和析构函数。  

#### 函数原型

```
TensorDesc();
~TensorDesc() = default;
explicit TensorDesc(Shape shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT);
TensorDesc(const TensorDesc &desc);
TensorDesc(TensorDesc &&desc);
TensorDesc &operator=(const TensorDesc &desc);
TensorDesc &operator=(TensorDesc &&desc);
```

#### 参数说明

|参数名|输入/输出|描述|
|:-----|:----|:-------------------------------------------------------------------------------------------------------------------------------------------|
|shape|输入|Shape对象。|
|format|输入|Format对象，默认取值FORMAT_ND。 关于Format数据类型的定义，请参见[Format](./cannkit-ge-format.md)。|
|dt|输入|DataType对象，默认取值DT_FLOAT。 关于DataType数据类型的定义，请参见[DataType](./cannkit-ge-datatype.md)。|
|desc|输入|待拷贝或者移动的TensorDesc对象。|

#### 返回值

TensorDesc构造函数返回TensorDesc类型的对象。  

#### 异常处理

无  

#### 约束说明

无  

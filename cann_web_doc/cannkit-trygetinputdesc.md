# TryGetInputDesc

#### 函数功能

根据算子Input名称获取算子Input的TensorDesc。  

#### 函数原型

![](https://media:401788444115566936)  
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus TryGetInputDesc(const std::string &name, TensorDesc &tensor_desc) const;
graphStatus TryGetInputDesc(const char_t *name, TensorDesc &tensor_desc) const;
```

#### 参数说明

|参数名|输入/输出|描述|
|:----------|:----|:---------------------------|
|name|输入|算子的Input名。|
|tensor_desc|输出|返回算子端口的当前设置格式，为TensorDesc对象。|

#### 返回值

|类型|描述|
|:----------|:------------------------------------------------------------------------|
|graphStatus|GRAPH_SUCCESS：有此端口，获取TensorDesc成功。 GRAPH_FAILED：无此端口，出参为空，获取TensorDesc失败。|

#### 异常处理

|异常场景|说明|
|:--------|:--------------|
|无对应name输入|返回GRAPH_FAILED。|

#### 约束说明

无  

# UpdateDynamicOutputDesc

#### 函数功能

根据name和index的组合更新算子动态Output的TensorDesc。  

#### 函数原型

![](https://media:401788444115772940)  
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus UpdateDynamicOutputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc);
graphStatus UpdateDynamicOutputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc);
```

#### 参数说明

|参数名|输入/输出|描述|
|:----------|:----|:-------------|
|name|输入|算子动态Output的名称。|
|index|输入|算子动态Output编号。|
|tensor_desc|输入|TensorDesc对象。|

#### 返回值

|类型|描述|
|:----------|:-----------------------------------------------|
|graphStatus|更新动态Output成功，返回GRAPH_SUCCESS， 否则，返回GRAPH_FAILED。|

#### 异常处理

无  

#### 约束说明

无  

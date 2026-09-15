# CreateOperator

#### 函数功能

基于算子名称和算子类型获取算子对象实例。  

#### 函数原型

![](https://media:401788444102209870)  
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type)
static Operator CreateOperator(const char_t *const operator_name, const char_t *const operator_type)
```

#### 参数说明

|参数名|输入/输出|描述|
|:------------|:----|:----|
|operator_name|输入|算子名称。|
|operator_type|输入|算子类型。|

#### 返回值

|类型|描述|
|:-----|:------|
|string|算子对象实例。|

#### 约束说明

无  

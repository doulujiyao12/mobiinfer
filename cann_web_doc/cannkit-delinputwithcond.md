# DelInputWithCond

#### 函数功能

根据算子属性，删除算子指定输入边。  

#### 函数原型

![](https://media:401788444112530918)  
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
OpRegistrationData &DelInputWithCond(int32_t inputIdx, const std::string &attrName, bool attrValue);
OpRegistrationData &DelInputWithCond(int32_t input_idx, const char_t *attr_name, bool attr_value);
```

#### 参数说明

|参数|输入/输出|说明|
|:--------|:----|:----------|
|inputIdx|输入|需要删除的输入边编号。|
|attrName|输入|属性名字。|
|attrValue|输入|属性的值。|

#### 约束说明

无  

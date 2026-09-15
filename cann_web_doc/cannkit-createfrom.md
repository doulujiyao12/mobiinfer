# CreateFrom

#### 函数功能

将传入的DT类型（支持int64_t、float、std::string类型）的参数转换为对应T类型（支持INT、FLOAT、STR类型）的参数。

* 支持将int64_t类型转换为INT类型。

* 支持将float类型转换为FLOAT类型。

* 支持将std::string类型转换为STR类型。

#### 函数原型

```
template<typename T, typename DT>
static T CreateFrom(DT &&val)
```

#### 参数说明

|参数名|输入/输出|描述|
|:--|:----|:-------|
|val|输入|DT类型的参数。|

#### 返回值

|类型|描述|
|:------------------|:------|
|- INT - FLOAT - STR|T类型的参数。|

#### 异常处理

无  

#### 约束说明

无  

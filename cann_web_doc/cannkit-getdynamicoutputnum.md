# GetDynamicOutputNum

#### 函数功能

获取算子的动态Output的实际个数。  

#### 函数原型

![](https://media:401788444111763914)  
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
int32_t GetDynamicOutputNum(const std::string &name) const;
int32_t GetDynamicOutputNum(const char_t *name) const;
```

#### 参数说明

|参数名|输入/输出|描述|
|:---|:----|:------------|
|name|输入|算子的动态Output名。|

#### 返回值

|类型|描述|
|:--|:-----------------------------------------|
|int|实际动态Output的个数。 当name非法，或者算子无动态Output时，返回0。|

#### 约束说明

无  

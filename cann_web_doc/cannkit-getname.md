# GetName

#### 函数功能

获取算子名称。  

#### 函数原型

![](https://media:401788444113840927)  
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
std::string GetName() const;
graphStatus GetName(AscendString &name) const;
```

#### 参数说明

|参数名|输入/输出|描述|
|:---|:----|:----|
|name|输出|算子名称。|

#### 返回值

|类型|描述|
|:----------|:---------------------------------|
|graphStatus|GRAPH_FAILED：失败。 GRAPH_SUCCESS：成功。|

#### 异常处理

无  

#### 约束说明

无  

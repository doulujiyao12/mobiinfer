# GetExpandDimsRule

#### 函数功能

获取Tensor的补维规则。  

#### 函数原型

```
graphStatus GetExpandDimsRule(AscendString &expand_dims_rule) const;
```

#### 参数说明

|参数名|输入/输出|描述|
|:---------------|:----|:-------------|
|expand_dims_rule|引用输入|获取到的补维规则，作为出参。|

#### 返回值

|类型|描述|
|:----------|:-------------------------------------|
|graphStatus|获取成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。|

#### 异常处理

无  

#### 约束说明

无  

# IsValid

#### 函数功能

判断Tensor对象是否有效。

若实际Tensor数据的大小与TensorDesc所描述的Tensor数据大小一致，则有效。  

#### 函数原型

```
graphStatus IsValid()
```

#### 参数说明

无  

#### 返回值

|类型|描述|
|:----------|:-----------------------------------------------|
|graphStatus|如果Tensor对象有效，则返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。|

#### 异常处理

无  

#### 约束说明

无  

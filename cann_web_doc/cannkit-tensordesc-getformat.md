# GetFormat

#### 函数功能

获取TensorDesc所描述的Tensor的Format。  

#### 函数原型

```
Format GetFormat() const;
```

#### 参数说明

无  

#### 返回值

|类型|描述|
|:----------------------------------------------------------------------------------------|:-----------------------------|
|[Format](./cannkit-ge-format.md)|TensorDesc所描述的Tensor的format信息。|

#### 异常处理

无  

#### 约束说明

由于返回的Format信息为值拷贝，因此修改返回的Format信息，不影响TensorDesc中已有的Format信息。  

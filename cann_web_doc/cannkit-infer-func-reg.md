# INFER_FUNC_REG

#### 函数功能

注册算子的InferShape函数。  

#### 函数原型

```
INFER_FUNC_REG(op_name, x)
```

该函数内部会自动调用INFER_VERIFY_FUNC(op_name, x)，INFER_VERIFY_FUNC函数中的op_name为算子的类型，x为指向INFER_FUNC_REG（op_name,x）中"x"的指针。  

#### 约束说明

无  

#### 参数说明

|参数名|输入/输出|描述|
|:------|:----|:---------------------------------------------------------------------------------------------------------------------------------------------|
|op_name|输入|算子类型。|
|x|输入|InferShape函数名，和[IMPLEMT_INFERFUNC](./cannkit-implemt-inferfunc.md)的InferShape函数名保持一致。|

#### 返回值

无  

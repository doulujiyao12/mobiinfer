# ValueDepend

#### 函数功能

标识该输入是否为"数据依赖输入"，数据依赖输入是指在Tiling/InferShape等函数实现时依赖该输入的具体数据。该输入数据为host侧数据，开发者在Tiling函数/InferShape函数中可以通过TilingContext类的[GetInputTensor](./cannkit-getinputtensor.md)/InferShapeContext类的[GetInputTensor](./cannkit-infershapecontext-getinputtensor.md)获取这个输入数据。  

#### 函数原型

```
OpParamDef &ValueDepend(Option value_depend);
OpParamDef &ValueDepend(Option value_depend, DependScope scope);
```

#### 参数说明

|参数|输入/输出|说明|
|:-----------|:----|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|value_depend|输入|value_depend有以下两种取值： - REQUIRED：表示算子的输入必须是Const类型。 会校验算子的输入是否是Const类型。若校验通过，则将此输入的值下发到算子，否则报错。 - OPTIONAL：表示算子的输入可以是Const类型，也可以不是Const类型。如果输入是Const类型，则将输入的值下发到算子，否则不下发。|
|scope|输入|scope类型为枚举DependScope，支持的取值为： - ALL：指在Tiling/InferShape等函数实现时都依赖该输入的具体数据，行为与调用单参数的ValueDepend接口一致。 - TILING：指仅在Tiling时依赖Tensor的值，可以支持Tiling下沉。|

#### 返回值

[OpParamDef](./cannkit-paramtype.md)算子定义。  

#### 约束说明

仅支持对算子输入配置，且仅支持输入的[DataType](./cannkit-ge-datatype.md)配置为DT_INT64/DT_FLOAT/DT_BOOL。  

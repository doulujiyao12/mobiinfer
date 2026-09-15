# InferShapeRange

#### 函数功能

注册算子的InferShapeRange函数。

开发者需要为算子编写一个InferShapeRangeKernelFunc类型的函数，并使用该接口进行注册。

InferShapeRangeKernelFunc类型定义如下。

```
using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext *);
```

#### 函数原型

```
OpImplRegisterV2 &InferShapeRange(InferShapeRangeKernelFunc infer_shape_range_func);
```

#### 参数说明

|参数|输入/输出|说明|
|:---------------------|:----|:------------------------------------------------------------|
|infer_shape_range_func|输入|要注册的自定义infer_shape_range_func函数，类型为InferShapeRangeKernelFunc。|

#### 返回值

返回算子的OpImplRegisterV2对象，该对象新增注册了InferShapeRange函数infer_shape_range_func。  

#### 约束说明

无  

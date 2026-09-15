# GetPlacement

#### 函数功能

获取tensor的placement。  

#### 函数原型

```
TensorPlacement GetPlacement() const
```

#### 参数说明

无  

#### 返回值

返回tensor的placement。

关于TensorPlacement类型的定义，请参见[TensorPlacement](./cannkit-tensorplacement.md)。  

#### 约束说明

无  

#### 调用示例

```
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
              kFollowing, // placement
              ge::DT_FLOAT16, // dt
              nullptr};
auto placement = tensor.GetPlacement(); // kFollowing
```


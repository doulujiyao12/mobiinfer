# SetPlacement

#### 函数功能

设置tensor的placement。  

#### 函数原型

```
void SetPlacement(const TensorPlacement placement)
```

#### 参数说明

|参数|输入/输出|说明|
|:--------|:----|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
|placement|输入|需要设置的tensor的placement。 关于TensorPlacement类型的定义，请参见[TensorPlacement](./cannkit-tensorplacement.md)。|

#### 返回值

无  

#### 约束说明

无  

#### 调用示例

```
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
              kFollowing, // placement
              ge::DT_FLOAT16, // dt
              nullptr};
tensor.SetPlacement(TensorPlacement::kOnDeviceHbm);
auto placement = tensor.GetPlacement(); // kOnDeviceHbm
```


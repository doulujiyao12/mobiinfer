# GetExpandDimsType

#### 函数功能

获取shape的补维规则。  

#### 函数原型

```
ExpandDimsType GetExpandDimsType() const
```

#### 参数说明

无  

#### 返回值

返回shape的补维规则。

关于ExpandDimsType类型的定义，请参见[ExpandDimsType](./cannkit-expanddimstype-introduction.md)。  

#### 约束说明

无  

#### 调用示例

```
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
              kFollowing, // placement
              ge::DT_FLOAT16, // dt
              nullptr};
auto expand_dims_type = tensor.GetExpandDimsType(); // ExpandDimsType{}
```


# GetInputTensor

#### 函数功能

根据算子输入索引获取对应的输入tensor指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。  

#### 函数原型

```
const Tensor *GetInputTensor(const size_t index) const;
```

#### 参数说明

|参数|输入/输出|说明|
|:----|:----|:-------------|
|index|输入|算子输入索引，从0开始计数。|

#### 返回值

指定的输入tensor指针，当输入index非法时返回空指针。

关于Tensor类型的定义，请参见[Tensor](./cannkit-tensor-constructor.md)。  

#### 约束说明

如果输入没有被设置为数据依赖，调用此接口获取tensor时，只能在tensor中获取到正确的shape、format、datatype信息，无法获取到真实的tensor数据地址（获取到的地址为nullptr）。  

#### 调用示例

```
ge::graphStatus Tiling4ReduceCommon(TilingContext* context) {
  auto in_shape = context->GetInputShape(0);
  GE_ASSERT_NOTNULL(in_shape);
  auto axes_tensor = context->GetInputTensor(1);
  // ...
}
```


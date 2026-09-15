# GetInputTensorRange

#### 函数功能

根据算子输入索引获取对应的输入tensor range指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。  

#### 函数原型

```
using TensorRange = Range<Tensor> 
const TensorRange *GetInputTensorRange(const size_t index) const;
```

#### 参数说明

|参数|输入/输出|说明|
|:----|:----|:-------------|
|index|输入|算子输入索引，从0开始计数。|

#### 返回值

输入tensor range的指针，index非法时，返回空指针。  

#### 约束说明

如果输入没有被设置为数据依赖，调用此接口获取tensor range时，只能在tensor中获取到正确的shape、format、datatype信息，无法获取到真实的tensor数据地址（获取到的地址为nullptr）。  

#### 调用示例

```
const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
  auto input_shape_range = context->GetInputTensorRange(0U);
  auto output_shape_range = context->GetOutputShapeRange(0U);
  *output_shape_range->GetMin() = input_shape_range->GetMin()->GetStorageShape();
  *output_shape_range->GetMax() = input_shape_range->GetMax()->GetStorageShape();
  return GRAPH_SUCCESS;
};
```


# AutoMappingByOpFn

#### 函数功能

自动映射回调函数。  

#### 函数原型

```
Status AutoMappingByOpFn(const ge::Operator &op_src, ge::Operator &op);
```

#### 参数说明

|参数|输入/输出|说明|
|:-----|:----|:------------------------|
|op_src|输入|转换前原始模型中的算子，包含原始模型中算子的属性。|
|op|输入|适配AI处理器的算子。|

关于Operator类，请参见[Operator](./cannkit-operator-construction-and-destructor.md)。  

#### 调用示例

原始TensorFlow算子与适配AI处理器的算子属性一一映射的场景：

```
REGISTER_CUSTOM_OP("SoftplusGrad")
.FrameworkType(TENSORFLOW)
.OriginOpType("SoftplusGrad")
.ParseParamsByOperatorFn(AutoMappingByOpFn)
.ImplyType(ImplyType::TVM);
```

原始TensorFlow算子与适配AI处理器的算子属性无法一一映射的场景：

```
Status ParseResizeArea(const ge::Operator &op_src, ge::Operator& op)
  {
    AutoMappingByOpFn(op_src, op);
 
    ge::TensorDesc input_tensor = op.GetInputDesc("images");
    input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    input_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret = op.UpdateInputDesc("images", input_tensor);
    if(ret != ge::GRAPH_SUCCESS){
        return FAILED;
    }
    ge::TensorDesc output_tensor = op.GetOutputDesc("y");
    output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    output_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_output = op.UpdateOutputDesc("y", output_tensor);
    if(ret_output != ge::GRAPH_SUCCESS){
        return FAILED;
    }
    return SUCCESS;
  }
// 将ResizeArea操作注册到GE
REGISTER_CUSTOM_OP("ResizeArea")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("ResizeArea")
  .ParseParamsByOperatorFn(ParseResizeArea)
  .ImplyType(ImplyType::AI_CPU);
```


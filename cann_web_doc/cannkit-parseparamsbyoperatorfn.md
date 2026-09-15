# ParseParamsByOperatorFn

#### 函数功能

注册解析开发者自定义算子属性的函数。  

#### 函数原型

```
OpRegistrationData &ParseParamsByOperatorFn(const ParseParamByOpFunc &parse_param_by_op_fn)
```

#### 参数说明

|参数|输入/输出|说明|
|:-------------------|:----|:--------------------------------------------------------------------|
|parse_param_by_op_fn|输入|解析开发者自定义算子属性的函数，请参见[回调函数ParseParamByOpFunc](#回调函数parseparambyopfunc)。|

#### 回调函数ParseParamByOpFunc

开发者自定义并实现ParseParamByOpFunc类函数，完成原始模型中算子属性到适配AI处理器的模型中属性的映射，将结果填入Operator类中。

```
Status ParseParamByOpFunc(const ge::Operator &op_origin, ge::Operator &op_dest)
```

表1 参数说明  

|参数|输入/输出|说明|
|:--------|:----|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|op_origin|输入|框架定义的Operator类对象，包含解析出的原始模型中自定义算子属性信息，关于Operator类，请参见[Operator](./cannkit-operator-construction-and-destructor.md)。|
|op_dest|输出|适配AI处理器的模型中的算子数据结构，保存算子信息。 关于Operator类，请参见[Operator](./cannkit-operator-construction-and-destructor.md)。|

#### 约束说明

无  

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
// register ResizeArea op to GE
REGISTER_CUSTOM_OP("ResizeArea")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("ResizeArea")
  .ParseParamsByOperatorFn(ParseResizeArea)
  .ImplyType(ImplyType::AI_CPU);
```


# ParseOpToGraphFn

#### 函数功能

注册实现算子一对多子图映射的函数，即将算子映射为多个算子。  

#### 函数原型

```
OpRegistrationData &ParseOpToGraphFn(const ParseOpToGraphFunc &parse_op_to_graph_fn)
```

#### 参数说明

|参数|输入/输出|说明|
|:-------------------|:----|:-------------------------------------------------------------------------|
|parse_op_to_graph_fn|输入|实现算子一对多映射，进行子图构造的函数。 请参见[回调函数ParseOpToGraphFunc](#回调函数parseoptographfunc)。|

#### 约束说明

实现一对多子图映射时，插件注册时首先需要将原始框架中的算子映射成AI处理器中的PartitionedCall算子，并在[ParseParamsByOperatorFn](./cannkit-parseparamsbyoperatorfn.md)函数中使用"SetAttr"接口设置original_type。

实现样例请参见[调用示例](#调用示例)。  

#### 回调函数ParseOpToGraphFunc

开发者自定义并实现ParseOpToGraphFunc函数，通过IR模型构建方式完成一对多子图的构造。

回调函数原型定义如下。

```
Status  ParseOpToGraphFunc(const ge::Operator &op, ge::Graph &graph)
```

表1 参数说明  

|参数|输入/输出|说明|
|:----|:----|:---------------------------------|
|op|输入|PartitionedCall算子数据结构，Operator类对象。|
|graph|输出|构造的子图。|

子图输入输出关系构建方式如下。

* 输入：通过添加Data节点标识，Data节点的index属性表示原节点的第index个输入边。

* 输出：通过Graph::SetOutputs()接口设置，该接口的入参为std::vector\<std::pair\<Operator, std::vector\<size_t\>\>\> ，输出边按照设置的输出顺序相连。

#### 调用示例

以将Add算子转换成Addn+Abs为例。

实现Add算子到PartitionedCall算子的映射函数示例如下所示：

```
Status ParseParams(const ge::Operator &op_src, ge::Operator& op_dest)
{
    // ...
    op_dest.SetAttr("original_type", "ai.onnx::11::Add");
}
```

一对多子图构造函数实现示例如下所示：

```
static Status ParseOpToGraph(const Operator &op, Graph &graph) {
  auto data_0 = op::Data().set_attr_index(0);
  auto data_1 = op::Data().set_attr_index(1);
  auto addn = op::AddN("addn_sum").create_dynamic_input_x(2)
      .set_dynamic_input_x(0, data_0)
      .set_dynamic_input_x(1, data_1)
      .set_attr_N(2);
  auto abs = op::Abs("abs_sum").set_input_x(addn);
  std::vector<Operator> inputs{data_0, data_1};
  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(abs, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return domi::SUCCESS;
}
```

进行注册：

```
REGISTER_CUSTOM_OP("PartitionedCall")
.FrameworkType(xx)
.OriginOpType(xx)
.ParseParamsByOperatorFn(ParseParams)
.ParseOpToGraphFn(ParseOpToGraph)
.ImplyType(ImplyType::TVM);
```

图为将Add算子进行一对多子图映射后的示例。

图1 一对多转换示意图

![](https://media:401788444111816915)  

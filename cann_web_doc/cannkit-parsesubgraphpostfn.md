# ParseSubgraphPostFn

#### 函数功能

根据算子类型，注册算子的子图中输入输出节点跟算子的输入输出的对应关系函数实现。  

#### 函数原型

![](https://media:401788444111430913)  
数据类型为ParseSubgraphFunc的接口后续版本会废弃，建议使用数据类型为ParseSubgraphFuncV2的接口。

```
OpRegistrationData &ParseSubgraphPostFn(const ParseSubgraphFunc &subgraph_post_fn)
OpRegistrationData &ParseSubgraphPostFn(const ParseSubgraphFuncV2 &subgraph_post_fn);
```

#### 参数说明

|参数|输入/输出|说明|
|:---------------|:----|:----------------------------------------------------------------------------------|
|subgraph_post_fn|输入|子图中输入输出节点跟算子的输入输出的对应关系函数对象。 详见[回调函数ParseSubgraphFuncV2](#回调函数parsesubgraphfuncv2) 。|

#### 约束说明

无  

#### 回调函数ParseSubgraphFuncV2

开发者自定义并实现ParseSubgraphFuncV2函数，完成解析子图中输入输出节点跟算子的输入输出的对应关系功能，回调函数原型定义如下。

```
Status ParseSubgraphFuncV2(const ge::AscendString &subgraph_name, const ge::Graph &graph)
```

表1 参数说明  

|参数|输入/输出|说明|
|:------------|:----|:-----|
|subgraph_name|输入|子图名字。|
|graph|输出|构造的子图。|


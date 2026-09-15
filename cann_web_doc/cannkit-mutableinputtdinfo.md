# MutableInputTdInfo

#### 函数功能

根据输入索引信息，获取算子的对应输入Tensor描述，注意，编译时无法确定的shape信息不在Tensor描述中（由于编译时无法确定shape，因此该Tensor描述里不包含shape信息）。  

#### 函数原型

```
CompileTimeTensorDesc *MutableInputTdInfo(const size_t index)
```

#### 参数说明

|参数|输入/输出|说明|
|:----|:----|:----------------|
|index|输入|算子输入对应的索引，从0开始计数。|

#### 返回值

返回Tensor描述。返回对象为非const。  

#### 约束说明

无  

#### 调用示例

```
  for (size_t i = 0; i < compute_node_info.GetInputsNum(); ++i) {
    auto td = compute_node_info.MutableInputTdInfo(i);
    // ...
  }
```


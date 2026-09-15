# GetAttrs

#### 函数功能

获取算子的属性值，仅在算子IR原型定义和调用IMPL_OP宏注册的属性值会被返回，其他属性值被丢弃。  

#### 函数原型

```
const RuntimeAttrs *GetAttrs() const
```

#### 参数说明

无  

#### 返回值

所有IR原型定义的属性值以及通过IMPL_OP宏注册的属性值，为const类型的对象，属性值按照IR原型定义的顺序依次保存。  

#### 约束说明

无  

#### 调用示例

```
auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
ASSERT_NE(ret, nullptr);
auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
auto attrs = compute_node_info->GetAttrs();
```


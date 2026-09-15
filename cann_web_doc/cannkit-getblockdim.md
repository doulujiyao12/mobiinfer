# GetBlockDim

#### 函数功能

获取blockDim，即参与计算的Vector或者Cube核数。blockDim的详细概念和设置方式请参考[SetBlockDim](./cannkit-setblockdim.md)。  

#### 函数原型

```
uint32_t GetBlockDim() const;
```

#### 参数说明

无  

#### 返回值

返回blockDim。  

#### 约束说明

无  

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto block_dim = context->GetBlockDim();
  // ...
}
```


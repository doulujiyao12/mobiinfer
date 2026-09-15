# GetRawTilingData

#### 函数功能

获取无类型的tiling data指针。  

#### 函数原型

```
TilingData *GetRawTilingData();
```

#### 参数说明

无  

#### 返回值

tiling data指针，失败时返回空指针。  

#### 约束说明

无  

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto tiling_data = context->GetRawTilingData();
  // ...
}
```


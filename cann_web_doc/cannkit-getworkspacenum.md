# GetWorkspaceNum

#### 函数功能

获取workspace个数。  

#### 函数原型

```
size_t GetWorkspaceNum() const;
```

#### 参数说明

无  

#### 返回值

workspace的个数。  

#### 约束说明

无  

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto ws_num = context->GetWorkspaceNum();
  // ...
}
```


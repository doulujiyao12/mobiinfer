# GetTilingCond

#### 函数功能

获取tiling cond。  

#### 函数原型

```
int32_t GetTilingCond() const;
```

#### 参数说明

无  

#### 返回值

tiling cond:

若返回值大于等于0，代表此tiling cond为有效的tiling cond。

若返回值为-1，代表此tiling cond为无效的tiling cond。  

#### 约束说明

无  

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto tiling_cond = context->GetTilingCond();
  // ...
}
```


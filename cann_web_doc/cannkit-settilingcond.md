# SetTilingCond

#### 函数功能

设置tiling cond。  

#### 函数原型

```
ge::graphStatus SetTilingCond(int32_t tiling_cond);
```

#### 参数说明

|参数|输入/输出|说明|
|:----------|:----|:----------------|
|tiling_cond|输入|需要设置的tiling cond。|

#### 返回值

设置成功时返回"ge::GRAPH_SUCCESS"。

关于graphStatus的定义，请参见[ge::graphStatus](./cannkit-gegraphstatus.md)。  

#### 约束说明

当前支持的Kirin9020、Kirin9030和KirinX90系列处理器是分离架构。  

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto ret = context->SetTilingCond(10);
  // ...
}
```


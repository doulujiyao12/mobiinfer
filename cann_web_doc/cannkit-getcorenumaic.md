# GetCoreNumAic

#### 函数功能

获取当前硬件平台AI Core中Cube核数。若AI Core的架构为Cube、Vector分离架构，返回AI Core上的Cube核数；非分离架构返回AI Core的核数。  

#### 函数原型

```
uint32_t GetCoreNumAic(void) const;
```

#### 参数说明

无  

#### 返回值

若AI Core的架构为Cube、Vector分离架构，返回AI Core上的Cube核数；非分离架构返回AI Core的核数。

目前涉及分离架构的处理器包括：Kirin9020系列处理器、Kirin9030系列处理器。  

#### 约束说明

无  

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    // ...按照aivNum切分
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    return ret;
}
```


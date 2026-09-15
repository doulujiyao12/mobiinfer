# GetSocVersion

#### 函数功能

获取当前硬件平台版本型号。  

#### 函数原型

```
SocVersion GetSocVersion(void) const;
```

#### 参数说明

无  

#### 返回值

当前硬件平台版本型号的枚举类。该枚举类和AI处理器型号的对应关系请通过CANN DDK包里的ddk/ai_ddk_lib/include/tiling/platform/platform_ascendc.h头文件获取。  

#### 约束说明

无  

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    // 根据所获得的版本型号自行设计Tiling策略
    if (socVersion == platform_ascendc::SocVersion::KIRIN9020) {
        // ...
    }
    return ret;
}
```


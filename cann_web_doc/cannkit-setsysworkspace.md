# SetSysWorkSpace

#### 功能说明

框架需要使用的workspace称之为系统workspace。[Matmul](./cannkit-matmul-usage-description.md)等高阶API需要系统workspace，所以在使用该类API时，需要调用该接口，设置系统workspace的指针。

在kernel侧调用该接口前，需要在host侧调用GetLibApiWorkSpaceSize获取系统workspace的大小，并在host侧设置workspacesize大小。样例如下。

```
// 开发者自定义的tiling函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddApiTiling tiling;
    // ...
    size_t usrSize = 256; // 设置开发者需要使用的workspace大小。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    // ...
}
```

#### 函数原型

```
__aicore__ inline void SetSysWorkSpace(GM_ADDR workspace)
```

#### 参数说明

表1 接口参数说明  

|参数名称|输入/输出|描述|
|:--------|:----|:------------------------------------------------|
|workspace|输入|核函数传入的workspace的指针，包括系统workspace和开发者使用的workspace。|

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

无  

#### 返回值

无  

#### 调用示例

```
template<typename aType, typename bType, typename cType, typename biasType> 
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, const TCubeTiling& tiling, float alpha)
{
    // 融合算子的初始化操作
    // ...
    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}
```


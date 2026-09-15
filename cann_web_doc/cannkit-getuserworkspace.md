# GetUserWorkspace

#### 功能说明

获取开发者使用的[workspace](./cannkit-getsysworkspaceptr.md)指针。如果使用了[Matmul](./cannkit-matmul-usage-description.md)等需要系统workspace的高阶API，kernel侧需要通过[SetSysWorkSpace](./cannkit-setsysworkspace.md)设置系统workspace，此时开发者workspace需要通过该接口获取。  

#### 函数原型

```
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
```

#### 参数说明

表1 接口参数说明  

|参数名称|输入/输出|描述|
|:--------|:----|:--------------------------------------------|
|workspace|输入|传入workspace的指针，包括系统workspace和开发者使用的workspace。|

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

无  

#### 返回值

开发者使用workspace指针。  

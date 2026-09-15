# GetBlockNum

#### 功能说明

获取当前任务配置的核数，用于代码内部的多核逻辑控制等。  

#### 函数原型

```
__aicore__ inline int64_t GetBlockNum()
```

#### 参数说明

无  

#### 返回值

当前任务配置的核数。  

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 约束说明

无  

#### 调用示例

```
#include "kernel_operator.h"
// 在核内做简单的tiling计算时使用block_num，复杂tiling建议在host侧完成
__aicore__ inline void InitTilingParam(int32_t& totalSize, int32_t& loopSize)
{
    loopSize = totalSize / AscendC::GetBlockNum();
};
```


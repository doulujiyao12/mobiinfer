# GetBlockIdx

#### 功能说明

获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。  

#### 函数原型

```
__aicore__ inline int64_t GetBlockIdx()
```

#### 参数说明

无  

#### 返回值

当前核的index，index的范围为\[0, 开发者配置的block_dim数量 - 1\]。  

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 约束说明

GetBlockIdx为一个系统内置函数，返回当前核的index。  

#### 调用示例

```
#include "kernel_operator.h"
constexpr int32_t SINGLE_CORE_OFFSET = 256;
class KernelGetBlockIdx {
public:
    __aicore__ inline KernelGetBlockIdx () {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        // 根据index对每个核进行地址偏移
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        src1Global.SetGlobalBuffer((__gm__ float*)src1Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        pipe.InitBuffer(inQueueSrc0, 1, 256 * sizeof(float));
        pipe.InitBuffer(inQueueSrc1, 1, 256 * sizeof(float));
        pipe.InitBuffer(selMask, 1, 256);
        pipe.InitBuffer(outQueueDst, 1, 256 * sizeof(float));
    }
    // ...
};
```


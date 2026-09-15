# 更多样例

#### 样例模板

为了方便开发者快速运行具体指令中的参考样例，本章节提供单目指令的样例模板。

开发者可以将以下样例模板作为代码框架，只需将具体指令中的样例片段拷贝替换下文代码段中的加粗内容即可。

```
#include "kernel_operator.h"
  
class KernelUnary {
public:
    __aicore__ inline KernelUnary() {}
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(half));
        pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, 512);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
  
        **AscendC::Sqrt(dstLocal, srcLocal, 512);**
  
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
};
extern "C" __global__ __aicore__ void unary_simple_kernel(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
{
    KernelUnary op;
    op.Init(srcGm, dstGm);
    op.Process();
}
```


# HasTensorInQue

#### 功能说明

查询队列中目前是否已有入队的Tensor。  

#### 函数原型

```
__aicore__ inline bool HasTensorInQue()
```

#### 参数说明

无  

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

无  

#### 返回值

* true：表示Queue中存在已入队的Tensor。

* false：表示Queue为完全空闲。

#### 调用示例

```
// 根据HasTensorInQue判断当前Queue中是否有已入队的Tensor，当前Queue的深度为4，无内存EnQue动作，返回为false
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.HasTensorInQue();
```


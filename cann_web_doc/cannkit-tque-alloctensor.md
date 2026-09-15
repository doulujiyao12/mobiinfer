# AllocTensor

#### 功能说明

从队列中分配Tensor，Tensor所占大小为InitBuffer时设置的每块内存长度。注意，分配的Tensor内容并非全0，可能会是随机值。  

#### 函数原型

```
template <typename T> 
__aicore__ inline LocalTensor<T> AllocTensor()
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

LocalTensor对象。  

#### 调用示例

```
// 使用AllocTensor分配Tensor
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len); // InitBuffer分配内存块数为4，每块大小为1024Bytes
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>(); // AllocTensor分配Tensor长度为1024Bytes
```


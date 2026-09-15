# FreeTensor

#### 功能说明

释放队列中的指定Tensor，供Que后续使用。  

#### 函数原型

```
template <typename T> 
__aicore__ inline void FreeTensor(LocalTensor<T>& tensor)
```

#### 参数说明

|参数名称|输入/输出|含义|
|:-----|:----|:----------|
|tensor|输入|待释放的Tensor。|

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
// 使用FreeTensor释放通过AllocTensor分配的Tensor，注意配对使用
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.FreeTensor<half>(tensor1);
```


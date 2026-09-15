# Get

#### 功能说明

从TBuf上获取指定长度的Tensor，或者获取全部长度的Tensor。  

#### 函数原型

* 获取全部长度的Tensor

  ```
  LocalTensor<T> Get<T>()
  ```

* 获取指定长度的Tensor

  ```
  LocalTensor<T> Get<T>(uint32_t len)
  ```

#### 参数说明

表1 参数说明  

|参数名称|输入/输出|含义|
|:---|:----|:---------------|
|len|输入|需要获取的Tensor元素个数。|

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

len的数值是Tensor中元素的个数，len\*sizeof(T)不能超过TBuf初始化时的长度。  

#### 返回值

无  

#### 调用示例

```
// 为TBuf初始化分配内存，分配内存长度为1024Bytes
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf; // 模板参数为TPosition中的VECCALC类型
uint32_t byteLen = 1024;
pipe.InitBuffer(calcBuf, byteLen);
// 从calcBuf获取Tensor，Tensor为pipe分配的所有内存大小，为1024Bytes
AscendC::LocalTensor<int32_t> tempTensor1 = calcBuf.Get<int32_t>();
// 从calcBuf获取Tensor，Tensor为128个int32_t类型元素的内存大小，为512Bytes
AscendC::LocalTensor<int32_t> tempTensor2 = calcBuf.Get<int32_t>(128);
```


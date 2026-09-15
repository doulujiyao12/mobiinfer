# 构造函数与析构函数

#### 函数功能

构造一个TensorData。

* 构造方式1：指定了tensor数据的地址以及用于管理tensor数据的函数manager。

* 构造方式2：指定了tensor数据的地址、用于管理tensor数据的函数manager、tensor数据所占内存大小、tensor数据所在的位置（host、device）。

* 构造方式3：移动构造形式。

![](https://media:401788444101931869)  
若manager为nullptr，则认为addr就是tensor的数据地址。否则，tensor数据的地址由manager给出。  

#### 函数原型

* 构造函数

  ```
  TensorData(TensorAddress addr = nullptr, const TensorAddrManager manager = nullptr)
  TensorData(TensorAddress addr, const TensorAddrManager manager, size_t size, TensorPlacement placement)
  TensorData(TensorData &&other) noexcept
  TensorData(const TensorData &) = delete;
  ```

* 析构函数

  ```
  ~TensorData()
  ```

#### 参数说明

|参数|输入/输出|说明|
|:--------|:----|:-----------------------------------------------------------|
|addr|输入|tensor数据的地址，定义如下。 using TensorAddress = void \*;|
|manager|输入|tensor data的管理函数，若manager为空，则认为addr就是tensor的数据地址，且此数据不需要被释放。|
|size|输入|tensor数据所占的内存大小。|
|placement|输入|tensor数据所在的设备位置。|

#### 返回值

初始化后的TensorData对象。  

#### 约束说明

无  

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
```


# SetSize

#### 函数功能

设置tensor数据的内存大小。  

#### 函数原型

```
void SetSize(const size_t size)
```

#### 参数说明

|参数|输入/输出|说明|
|:---|:----|:-----------------|
|size|输入|tensor的内存大小，单位为字节。|

#### 返回值

无  

#### 约束说明

无  

#### 调用示例

```
std::vector<int> a = {10};
auto addr = reinterpret_cast<void *>(a.data());
TensorData td(addr, HostAddrManager, 100U, kOnHost);
td.SetSize(10U);
```


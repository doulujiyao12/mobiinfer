# Init

#### 函数功能

初始化TilingData。  

#### 函数原型

```
void Init(const size_t cap_size, void *const data);
```

#### 参数说明

|参数|输入/输出|说明|
|:-------|:----|:--------------|
|cap_size|输入|最大容量，单位为字节。|
|data|输入|tiling data的地址。|

#### 返回值

无  

#### 约束说明

无  

#### 调用示例

```
size_t cap_size = 100U;
size_t total_size = cap_size + sizeof(TilingData);
auto td_buf = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]());
auto td = reinterpret_cast<TilingData *>(td_buf.get());
td->Init(cap_size, td_buf.get() + sizeof(TilingData)); // 内存平铺
```


# Init

#### 函数功能

使用最大容量初始化本实例。  

#### 函数原型

```
void Init(size_t capacity)
```

#### 参数说明

|参数|输入/输出|说明|
|:-------|:----|:---------|
|capacity|输入|初始化本实例的容量。|

#### 返回值

无  

#### 约束说明

无  

#### 调用示例

```
size_t capacity = 100U;
size_t total_size = capacity * sizeof(int64_t) + sizeof(ContinuousVector);
auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
reinterpret_cast<ContinuousVector *>(holder.get())->Init(capacity); // 100U
```


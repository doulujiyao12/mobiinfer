# GetSize

#### 函数功能

获取Tensor数据的内存大小。  

#### 函数原型

```
size_t GetSize() const
```

#### 参数说明

无  

#### 返回值

内存大小，单位是字节。  

#### 约束说明

无  

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {}, {}, ge::DT_FLOAT, nullptr};
auto td_size = t.GetSize(); // 1*2*3*sizeof(float) = 24;
```


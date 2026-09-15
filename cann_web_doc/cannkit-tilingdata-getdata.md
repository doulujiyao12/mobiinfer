# GetData

#### 函数功能

获取TilingData的数据指针。  

#### 函数原型

```
void *GetData();
const void *GetData() const;
```

#### 参数说明

无  

#### 返回值

data指针。  

#### 约束说明

无  

#### 调用示例

```
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());
auto tiling_data_ptr = td->GetData(); // td_buf.get() + sizeof(TilingData)
```


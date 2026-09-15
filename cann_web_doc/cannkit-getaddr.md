# GetAddr

#### 函数功能

获取tensor数据地址。若存在manager函数，则由manager函数给出地址。  

#### 函数原型

```
TensorAddress GetAddr() const
```

#### 参数说明

无  

#### 返回值

tensor地址。  

#### 约束说明

无  

#### 调用示例

```
auto addr0 = reinterpret_cast<void *>(0x10);
TensorData td(addr0, nullptr);
auto addr1 = td.GetAddr(); // 0x10
```


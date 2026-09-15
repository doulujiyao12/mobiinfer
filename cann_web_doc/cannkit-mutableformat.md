# MutableFormat

#### 函数功能

获取Tensor的format，包含运行时format和原始format。  

#### 函数原型

```
StorageFormat &MutableFormat()
```

#### 参数说明

无  

#### 返回值

format引用。

关于StorageFormat类型的定义，请参见[StorageFormat](./cannkit-storageformat-constructor.md)。  

#### 约束说明

无  

#### 调用示例

```
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
              kFollowing, // placement
              ge::DT_FLOAT16, // dt
              nullptr};
auto fmt = tensor.MutableFormat();
```


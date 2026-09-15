# GetStorageShape

#### 函数功能

获取运行时shape。  

#### 函数原型

```
const Shape &GetStorageShape() const
```

#### 参数说明

无  

#### 返回值

运行时shape。  

#### 约束说明

无  

#### 调用示例

```
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto storage_shape = shape.GetStorageShape(); // 256,256,3
```


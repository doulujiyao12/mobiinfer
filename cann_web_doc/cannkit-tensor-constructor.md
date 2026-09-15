# 构造函数

#### 函数功能

用于构造指定的Tensor对象。Tensor类用来描述一个tensor对象的信息以及行为，包含：shape信息、format信息、datatype信息以及tensor数据内容tensordata。  

#### 函数原型

```
Tensor (const StorageShape &storage_shape, const StorageFormat &storage_format, const TensorPlacement placement, const ge::DataType data_type, TensorAddress addr)
Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format, ge::DataType data_type)
```

#### 参数说明

|参数|输入/输出|说明|
|:-------------|:----|:-------------------------|
|storage_shape|输入|指定tensor的shape信息。|
|storage_format|输入|指定tensor的format信息。|
|placement|输入|指定tensor的实际数据所存储的device位置。|
|data_type|输入|指定tensor的datatype信息。|
|addr|输入|指定tensor的实际数据所存储的内存地址。|

#### 返回值

返回一个初始化的Tensor对象。  

#### 约束说明

无  

#### 调用示例

```
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
              kOnDeviceHbm, // placement
              ge::DT_FLOAT16, // dt
              nullptr};
```


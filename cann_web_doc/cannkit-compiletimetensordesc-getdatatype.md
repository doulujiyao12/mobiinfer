# GetDataType

#### 函数功能

获取CompileTimeTensorDesc所描述的Tensor的数据类型。  

#### 函数原型

```
ge::DataType GetDataType() const
```

#### 参数说明

无  

#### 返回值

DataType的声明[DataType](./cannkit-ge-datatype.md)。  

#### 约束说明

无  

#### 调用示例

```
auto dtype_ = ge::DataType::DT_INT32;
StorageFormat fmt_(ge::Format::FORMAT_NC, ge::FORMAT_NCHW, {});
ExpandDimsType type_("1001");
gert::CompileTimeTensorDesc td;
td.SetDataType(dtype_);
auto dtype = td.GetDataType(); // ge::DataType::DT_INT32;
td.SetStorageFormat(fmt_.GetStorageFormat());
auto storage_fmt = td.GetStorageFormat(); // ge::FORMAT_NCHW
td.SetOriginFormat(fmt_.GetOriginFormat());
auto origin_fmt = td.GetOriginFormat(); // ge::Format::FORMAT_NC
td.SetExpandDimsType(type_);
auto type = td.GetExpandDimsType(); // type_("1001")
```


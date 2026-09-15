# GetFormat

#### 函数功能

获取CompileTimeTensorDesc所描述的Tensor的数据排布格式。  

#### 函数原型

```
const StorageFormat &GetFormat() const
```

#### 参数说明

无  

#### 返回值

返回数据排布格式。  

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
td.SetStorageFormat(fmt_.GetFormat());
auto storage_fmt = td.GetFormat(); // ge::FORMAT_NCHW
td.SetOriginFormat(fmt_.GetOriginFormat());
auto origin_fmt = td.GetOriginFormat(); // ge::Format::FORMAT_NC
td.SetExpandDimsType(type_);
auto type = td.GetExpandDimsType(); // type_("1001")
```


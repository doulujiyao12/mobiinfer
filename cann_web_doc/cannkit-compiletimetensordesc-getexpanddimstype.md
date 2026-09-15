# GetExpandDimsType

#### 函数功能

获取原始Format向运行时Format转换时的补维规则。  

#### 函数原型

```
ExpandDimsType GetExpandDimsType() const
```

#### 参数说明

无  

#### 返回值

补维规则，请参考[ExpandDimsType](./cannkit-expanddimstype-introduction.md)。  

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


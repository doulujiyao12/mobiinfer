# GetOriginFormat

#### 函数功能

获取Tensor的原始format。  

#### 函数原型

```
ge::Format GetOriginFormat() const
```

#### 参数说明

无  

#### 返回值

原始format。

关于ge::Format类型的定义，请参见[Format](./cannkit-ge-format.md)。  

#### 约束说明

无  

#### 调用示例

```
Tensor t = {{}, {}, {}, {}, nullptr};
t.SetOriginFormat(ge::FORMAT_NHWC);
t.SetStorageFormat(ge::FORMAT_NC1HWC0);
auto fmt = t.GetOriginFormat(); // ge::FORMAT_NHWC
```


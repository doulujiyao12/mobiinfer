# GetKernelType

#### 函数功能

获取当前内核的类型。  

#### 函数原型

```
const char *GetKernelType() const
```

#### 参数说明

无  

#### 返回值

当前内核的类型。  

#### 约束说明

无  

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto kernel_type = extend_context->GetKernelType();
```


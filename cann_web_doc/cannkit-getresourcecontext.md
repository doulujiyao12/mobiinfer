# GetResourceContext

#### 函数功能

通过资源标识key来获取对应的资源上下文对象。  

#### 函数原型

```
ResourceContext *GetResourceContext(const ge::AscendString &key)
```

#### 参数说明

|参数名|输入/输出|描述|
|:--|:----|:----------------------------------------------------------------------------------------------------------------|
|key|输入|资源的唯一标识。由资源类算子[InferShape](./cannkit-infershape.md)函数指定。|

#### 返回值

|类型|描述|
|:-----------------|:------------------------------------------------------------------------------------------------------------------|
|ResourceContext \*|资源上下文对象。 基础定义如下，资源类算子可以基于此扩展。 struct ResourceContext {virtual \~ResourceContext() {}}; 用于保存资源相关信息，如shape、datatype等。|

#### 约束说明

若使用[Create](./cannkit-create.md)接口创建InferenceContext时未传入resource context管理器指针，则该接口返回空指针，因此使用其返回值之前需要判空。  

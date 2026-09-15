# SetResourceContext

#### 函数功能

为标识为key的资源，设置资源上下文对象，并交由资源上下文管理器管理。

此接口一般由写类型的资源类算子调用，如stack push等。  

#### 函数原型

```
graphStatus SetResourceContext(const ge::AscendString &key, ResourceContext *resource_context)
```

#### 参数说明

|参数名|输入/输出|描述|
|:---------------|:----|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|key|输入|资源唯一标识。|
|resource_context|输入|资源上下文对象指针，可参见[GetResourceContext](./cannkit-getresourcecontext.md)接口的[返回值](./cannkit-getresourcecontext.md)。|

#### 返回值

graphStatus：GRAPH_SUCCESS，代表成功；GRAPH_FAILED，代表失败。  

#### 约束说明

若使用[Create](./cannkit-create.md)接口创建InferenceContext时未传入resource context管理器指针，则该接口返回失败。  

# GetCapacity

#### 函数功能

获取最大可保存的元素个数。  

#### 函数原型

```
size_t GetCapacity() const
```

#### 参数说明

无  

#### 返回值

最大可保存的元素个数。  

#### 约束说明

无  

#### 调用示例

```
size_t capacity = 100U;
auto cv_holder = ContinuousVector::Create<int64_t>(capacity);
auto cv = reinterpret_cast<ContinuousVector *>(cv_holder.get());
auto cap = cv->GetCapacity(); // 100U
```


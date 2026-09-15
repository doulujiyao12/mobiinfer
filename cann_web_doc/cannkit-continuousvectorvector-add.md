# Add

#### 函数功能

新增一个ContinuousVector元素，其中新增ContinuousVector元素的容量为inner_vector_capacity。  

#### 函数原型

```
template<typename T> ContinuousVector *Add(size_t inner_vector_capacity)
```

#### 参数说明

|参数|输入/输出|说明|
|:--------------------|:----|:-----------------------|
|inner_vector_capacity|输入|新增ContinuousVector元素的容量。|

#### 返回值

新增ContinuousVector元素的首地址。  

#### 约束说明

无  

#### 调用示例

```
// 创建ContinuousVectorVector对象cvv
// ...
// 增加元素
size_t inner_vector_capacity = 2;
auto cv = cvv->Add(inner_vector_capacity);
```


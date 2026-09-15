# GetFullSize

#### 函数功能

获取补维后的dim数。  

#### 函数原型

```
AxisIndex GetFullSize() const
```

#### 参数说明

无  

#### 返回值

返回补维规则的长度，或者说是补维规则描述的维度。  

#### 约束说明

无  

#### 调用示例

```
ExpandDimsType type1("1001");
auto dim_num = type1.GetFullSize(); // dim_num=4
```


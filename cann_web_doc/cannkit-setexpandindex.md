# SetExpandIndex

#### 函数功能

将第index轴设置为补维轴。  

#### 函数原型

```
void SetExpandIndex(const AxisIndex index)
```

#### 参数说明

|参数|输入/输出|说明|
|:----|:----|:----------------------------------------|
|index|输入|第index根轴为补维轴。 using AxisIndex = uint64_t;|

#### 返回值

无  

#### 约束说明

无  

#### 调用示例

```
ExpandDimsType type1("1001");
type1.SetExpandIndex(1); // 补维规则mask_=1101
```


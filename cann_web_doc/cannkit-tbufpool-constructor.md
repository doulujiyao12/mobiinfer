# 构造函数

#### 函数功能

创建TBufPool对象时，初始化数据成员。  

#### 函数原型

```
template <TPosition pos, uint32_t bufIDSize = 4> 
__aicore__ inline TBufPool();
```

#### 参数说明

|参数名称|含义|
|:--------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|pos|TBufPool逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。关于TPosition的具体介绍请参考[TPosition](./cannkit-tposition.md)。|
|bufIDSize|TBufPool可分配Buffer数量，默认为4，不超过16。|

#### 约束说明

无  

# ScalarGetCountOfValue

#### 功能说明

获取一个uint64_t类型数字的二进制中0或者1的个数。  

#### 函数原型

```
template <int countValue>  
__aicore__ inline int64_t ScalarGetCountOfValue(uint64_t valueIn)
```

#### 参数说明

表1 参数说明  

|参数名|输入/输出|描述|
|:---------|:----|:-------------------------------|
|valueIn|输入|被统计的二进制数字。 数据类型是uint64_t。|
|countValue|输入|指定统计0还是统计1的个数。 数据类型是int，只能输入0或1。|

#### 返回值

valueIn中0或者1的个数。  

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 约束说明

无  

#### 调用示例

```
uint64_t valueIn = 0xffff;
// 输出数据(oneCount): 16
int64_t oneCount = AscendC::ScalarGetCountOfValue<1>(valueIn);
```


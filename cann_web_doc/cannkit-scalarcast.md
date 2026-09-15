# ScalarCast

#### 功能说明

将一个scalar的类型转换为指定的类型。  

#### 函数原型

```
template <typename srcT, typename dstT, RoundMode roundMode> 
__aicore__ inline dstT ScalarCast(srcT valueIn)
```

#### 参数说明

表1 参数说明  

|参数名|输入/输出|描述|
|:--------|:----|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|valueIn|输入|被转换的scalar。|
|srcT|输入|valueIn的数据类型，支持float。|
|dstT|输入|转换后的数据类型，支持half、int32_t。|
|roundMode|输入|精度转换处理模式，类型是RoundMode。 RoundMode为枚举类型，用以控制精度转换处理模式，具体取值为：CAST_NONE、CAST_RINT、CAST_FLOOR、CAST_ROUND、CAST_TRUNC、CAST_ODD。 对于ScalarCast，转换类型仅支持float转half(f32 to f16)与float转int32_t(f32 to s32)，相应支持的RoundMode如下。 - f32 to f16：CAST_ODD - f32 to s32：CAST_ROUND、CAST_CEIL、CAST_FLOOR、CAST_RINT ScalarCast的精度转换规则与Cast保持一致，具体可参考[Cast函数功能](./cannkit-precision-conversion-instruction.md)下的表1。|

#### 返回值

dstT类型的valueIn。  

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 约束说明

无。  

#### 调用示例

```
float valueIn = 2.5;
// 输出数据(valueOut): 3
int32_t valueOut = AscendC::ScalarCast<float, int32_t, AscendC::RoundMode::CAST_ROUND>(valueIn);
```


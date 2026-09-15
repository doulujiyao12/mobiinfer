# 数据类型

KirinX90/Kirin9030除了不支持bfloat16_t，MrgSort、Mmad基础api（如不支持float等）、Cast基础API（如不支持s64）等部分API还存在其它数据类型差异，详见[《Ascend C算子接口》](./cannkit-ascend-c-apis.md)。若开发者在Ascend910B/Ascend910C使用bfloat16_t数据类型，在Kirin平台上需要替换成其它数据类型（比如half）。

表1 数据类型差异兼容说明  

|基础API|兼容说明|
|:--------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
|ToBfloat16、ToFloat|不支持功能和bfloat16_t数据类型绑定的接口|
|GatherMask、Cast、Duplicate、Brcb、Gather、Copy、DataCopy|不支持bfloat16_t数据类型|
|Sort32、MrgSort、Mmad|不支持float数据类型|
|SetHF32Mode、SetHF32TransMode|不支持HF32模式|
|InitConstValue|不支持bfloat16_t、float、int32_t、uint32_t数据类型|
|LoadData、Fixpipe|不支持bfloat16_t、float数据类型|
|Cast|详见[《Ascend C算子接口》](./cannkit-ascend-c-apis.md)|
|DataCopy: L1 Buffer-\>Bias Table|KirinX90/Kirin9030只支持fp16数据类型，且排布方式有差异，详见[数据搬运](./cannkit-basic-datacopy.md)。|

针对数据类型有差异的情况，建议开发者采用如下编程方式，支撑同一份算子代码在不同芯片平台执行：

* 方式1：将数据类型作为算子模板参数传入，参见[Tiling模板编程](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)。

* 方式2：通过适当的NPU_ARCH 编译宏隔离，在算子Kernel中使用using关键字实现数据类型参数化编程。以基础API Mmad为例。

```
    /* Step1: 参数化数据类型 */
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    using L0cDtype = float
#endif
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
    using L0cDtype = half
#endif
    // ...
    /* Step2： 各平台，归一化编程 */
    // kernel 分配内存大小举例
    tPipe->InitBuffer(l0cQue, tiling.dbL0c, tiling.baseM * tiling.baseN * sizeof(L0cDtype));
    // ...
    // kernel mmad计算api举例
    mmad<L0cDtype, L0aDtype, L0bDtype, BiasDtype>(dstLocal, fmLocal, filterLocal, biasLocal, mmadParams);
```

![](https://media:401788444096327827)  
* 各产品对应的NPU_ARCH，请参见[概述](./cannkit-migration-guidance-overview.md)中表1产品型号和__NPU_ARCH__的对应关系。

* 若数据类型差异影响算子输入和输出数据类型，建议采用方式1。

* 若只是算子中间的计算数据类型有差异，建议采用方式2。


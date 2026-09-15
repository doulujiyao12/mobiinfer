# 高阶API迁移指导

高阶API，数据类型支持范围存在差异，详见[《Ascend C算子接口》](./cannkit-ascend-c-apis.md)。数据类型差异兼容策略，参考[数据类型](./cannkit-basic-datatype.md)。下面重点描述接口功能差异的兼容说明。  

#### HCCL通信类

不支持HCCL通信类高阶API。  

#### 矩阵计算

KirinX90/Kirin9030支持Matmul高阶API，但在涉及领域特性的部分存在不兼容情况。

表1 Matmul高阶API  

|接口名称|兼容说明|
|:-------------|:------------|
|SetHF32|不支持。|
|SetSparseIndex|不支持。|
|MatmulConfig|不支持IBShare模板。|


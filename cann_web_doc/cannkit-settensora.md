# SetTensorA

#### 功能说明

设置矩阵乘的左矩阵A。  

#### 函数原型

```
__aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false)
__aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false)
__aicore__ inline void SetTensorA(SrcAT aScalar)
```

#### 参数说明

表1 参数说明  

|参数名|输入/输出|描述|
|:-----------|:----|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|gm|输入|A矩阵在Global Memory上的首地址。 Kirin9020系列处理器、Kirin9030系列处理器，支持的数据类型为：half。|
|leftMatrix|输入|A矩阵在TSCM上的首地址或者在VECOUT上的首地址。 Kirin9020系列处理器、Kirin9030系列处理器，支持的数据类型为：half。|
|aScalar|输入|A矩阵中设置的值。支持传入标量数据，标量数据会被扩展为一个形状为\[1, K\]的tensor参与矩阵乘计算，tensor的数值均为该标量值。例如，开发者可以通过将aScalar设置为1来实现矩阵B在K方向的reduce sum操作。 Kirin9020系列处理器、Kirin9030系列处理器，支持的数据类型为：half。|
|isTransposeA|输入|A矩阵是否需要转置。 说明： - 若A矩阵MatmulType ISTRANS参数设置为true，此参数可以为true也可以为false，即运行时可以转置和非转置交替使用。 - 若A矩阵MatmulType ISTRANS参数设置为false，此参数只能设置为false，若强行设置为true，精度会有异常。|

#### 返回值

无  

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器  

#### 注意事项

传入的TensorA地址空间大小需要保证不小于singleM \* singleK。  

#### 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a); // 设置左矩阵A
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```


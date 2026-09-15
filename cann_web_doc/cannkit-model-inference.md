# 模型推理

#### 基本概念

该场景是基本模型的使用场景，主要包含模型的编译和推理，其他场景是基础场景的一个扩展和功能增强。  

#### 业务流程

模型推理的主要开发流程如下图所示：

![](https://media:401788444063898593)  

#### 接口说明

以下接口为主要流程接口，如要使用更丰富的编译、加载和执行时的配置，请参见[API参考](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h)。

表1 CANN Kit模型推理相关接口功能介绍  

|接口名|描述|
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------|
|OH_NNCompilation\* OH_NNCompilation_ConstructWithOfflineModelBuffer(const void \*modelBuffer, size_t modelSize);|根据模型buffer创建模型编译实例。|
|OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation \*compilation, size_t deviceID);|设置模型编译和执行的目标设备。|
|OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation \*compilation);|执行模型编译，生成编译后的模型保存在compilation中。|
|OH_NNExecutor\* OH_NNExecutor_Construct(OH_NNCompilation \*compilation);|根据编译后的模型，创建模型推理的执行器。|
|NN_Tensor\* OH_NNTensor_Create(size_t deviceID, NN_TensorDesc\* tensorDesc);|构造输入输出Tensor。|
|OH_NN_ReturnCode OH_NNExecutor_RunSync(OH_NNExecutor \*executor, NN_Tensor \*inputTensor\[\], size_t inputCount, NN_Tensor \*outputTensor\[\], size_t outputCount);|执行模型的同步推理。|
|void OH_NNCompilation_Destroy(OH_NNCompilation \*\*compilation);|销毁模型编译实例。|
|OH_NN_ReturnCode OH_NNTensor_Destroy(NN_Tensor\*\* tensor);|销毁输入输出Tensor。|
|void OH_NNExecutor_Destroy(OH_NNExecutor \*\*executor);|销毁模型推理的执行器。|

#### 开发步骤

以下为模型推理的主要开发步骤，具体实现请参见[SampleCode](https://gitcode.com/HarmonyOS_Samples/cannkit-samplecode-clientdemo-cpp)。

1. 准备模型和开发环境。

   * 准备离线模型(OM模型)，可以通过tools_omg工具生成或从[Model Zoo](./cannkit-model-zoo.md)获取。
   * 下载并配置[DevEco Studio](https://developer.huawei.com/consumer/cn/deveco-studio/) 环境，确保可以正常开发和调试HarmonyOS应用。
2. [创建DevEco Studio项目](./cannkit-creating-a-project.md)。

3. 创建模型编译实例。

   调用[OH_NNCompilation_ConstructWithOfflineModelBuffer](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nncompilation_constructwithofflinemodelbuffer)读取模型buffer，创建模型编译实例。或者通过调用[OH_NNCompilation_ConstructWithOfflineModelFile](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nncompilation_constructwithofflinemodelfile)直接读取模型文件，创建模型编译实例。
4. 选择目标device。

   调用[OH_NNDevice_GetAllDevicesID](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nndevice_getalldevicesid)，获取所有的设备ID，查找name为"HIAI_F"字段的设备ID，记录并通过[OH_NNCompilation_SetDevice](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nncompilation_setdevice)设置到步骤3创建的编译实例中。
5. 执行模型编译。

   调用[OH_NNCompilation_Build](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nncompilation_build)，传入步骤3创建的模型编译实例，即可执行模型编译，编译后的模型数据仍然保存在模型编译实例中。
6. 创建模型执行器。

   调用[OH_NNExecutor_Construct](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nnexecutor_construct)，创建编译后模型对应的执行器实例。执行器创建完成后即可调用[OH_NNCompilation_Destroy](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nncompilation_destroy)销毁模型编译实例。
7. 构造输入输出Tensor。

   调用[OH_NNExecutor_GetInputCount](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nnexecutor_getinputcount)，查询输入的个数，通过[OH_NNExecutor_CreateInputTensorDesc](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nnexecutor_createinputtensordesc)获取到对应索引的TensorDesc，根据该TensorDesc通过[OH_NNTensor_Create](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nntensor_create)创建Tensor，即可向Tensor中写入实际数据。输出Tensor的构造与输入Tensor的构造过程一致。
8. 执行模型推理。

   调用[OH_NNExecutor_RunSync](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nnexecutor_runsync)，执行模型的同步推理功能，模型的输出数据保存在outputTensors中。开发者可根据需要对输出数据做相应的处理以得到期望的内容。
9. 销毁实例。

   * 调用[OH_NNExecutor_Destroy](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nnexecutor_destroy)，销毁创建的模型执行器实例。
* 调用[OH_NNTensor_Destroy](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h#oh_nntensor_destroy)，销毁创建的输入输出Tensor。  

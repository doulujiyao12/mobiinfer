# 模型转换前准备

CANN Kit当前仅支持Caffe、TensorFlow、ONNX和MindSpore模型转换为离线模型，其他格式的模型需要开发者自行转换为CANN Kit支持的模型格式。

1. 准备训练好的Caffe、TensorFlow、ONNX等模型。例如：[Caffe SqueezeNet V1.0](https://github.com/forresti/SqueezeNet)模型。

2. 下载[Tools](./cannkit-preparations.md)，解压使用Tools下的OMG工具，将TensorFlow或Caffe模型转换为IR模型，使用方式请参见[模型转换示例](./cannkit-model-conversion-example.md)。

![](https://media:401788444078370662)  
若TensorFlow或Caffe模型过大，可以在OMG转换之前使用[Tools下载](./cannkit-preparations.md)的轻量化工具，使用方式请参见[模型轻量化](./cannkit-lightweight-tool-overview.md)。  

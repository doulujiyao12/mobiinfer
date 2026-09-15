# SuperKernel

KirinX90/Kirin9030处理器不支持SuperKernel，所以如下接口在Kirin平台不生效。

表1 KirinX90/Kirin9030任务间同步API  

|基础API|兼容说明|
|:------------------------------|:----------------------------------------------------------------|
|SetNextTaskStart、WaitPreTaskEnd|不生效。 KirinX90/Kirin9030不支持SuperKernel特性，所以任务间同步API不生效。算子代码无需进行修改。|


# FetchEventID

#### 功能说明

根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。  

#### 函数原型

```
template <HardEvent evt> 
__aicore__ inline TEventID TPipe::FetchEventID()
__aicore__ inline TEventID TPipe::FetchEventID(HardEvent evt)
```

#### 参数说明

|参数名称|输入/输出|含义|
|:---|:----|:------------------|
|evt|输入|HardEvent类型，硬件同步类型。|

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

FetchEventID适用于获取TEventID后，立刻调用SetFlag、WaitFlag，即用于数据依赖的不同流水指令之间插入同步。  

#### 返回值

TEventID  

#### 调用示例

```
// 需要插入scalar与vector之间的同步，申请对应的HardEvent的ID
AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
```


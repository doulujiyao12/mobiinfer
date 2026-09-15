# GetBaseAddr

#### 功能说明

根据传入的logicPos（逻辑抽象位置），获取该位置的基础地址，只在CPU调试场景下此接口生效，在CPU调试中通常用于将Tensor地址由CPU地址转换为NPU地址。  

#### 函数原型

```
inline uint8_t* GetBaseAddr(int8_t logicPos)
```

#### 参数说明

|参数名称|输入/输出|含义|
|:-------|:----|:-------------------------------------------------------------------------------------------------------------|
|logicPos|输入|逻辑位置类型。该类型具体说明请参考[TPosition](./cannkit-tposition.md)。|

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

无  

#### 返回值

逻辑位置对应的基地址。  

#### 调用示例

```
auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(pos));
```


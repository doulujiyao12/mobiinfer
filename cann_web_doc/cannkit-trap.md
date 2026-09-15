# Trap

#### 函数功能

当软件产生异常后，使用该指令终止kernel运行。  

#### 函数原型

```
__aicore__ inline void Trap()
```

#### 参数说明

无  

#### 返回值

无  

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

该接口在kernel需要调试时使用。  

#### 调用示例

```
AscendC::Trap();
```


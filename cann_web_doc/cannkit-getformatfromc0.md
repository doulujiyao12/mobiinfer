# GetFormatFromC0

#### 函数功能

根据传入的format和c0format信息得到实际的format。

实际format为4字节大小，第1个字节的高四位为预留字段，低四位为c0 format，第2-3字节为子format信息，第4字节为主format信息，如下。

/\*

\* ---------------------------------------------------

\* \| 4 bits \| 4bits \| 2 bytes \| 1 byte \|

\* \|------------\|-------------\|----------------\|--------\|

\* \| reserved \| c0 -format \| sub-format \| format \|

\* ---------------------------------------------------

\*/  

#### 函数原型

```
inline int32_t GetFormatFromC0(int32_t format, int32_t c0_format)
```

#### 参数说明

|参数|输入/输出|说明|
|:--------|:----|:----------------------------------|
|format|输入|sub-format与主format信息，值不超过0xffffffU。|
|c0_format|输入|c0_format信息，值不超过0xfU。|

#### 返回值

指定的format和c0_format对应的实际format。  

#### 异常处理

无  

#### 约束说明

无  

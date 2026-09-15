# AscendStringToDataType

#### 函数功能

将DataType的字符串表达转换为DataType枚举值。

使用该接口需要包含type_utils.h头文件。

```
#include "graph/utils/type_utils.h"
```

#### 函数原型

```
static DataType AscendStringToDataType(const AscendString &str);
```

#### 参数说明

|参数|输入/输出|说明|
|:--|:----|:--------------------------------------------------------------------------------------------------------------------------------------------------|
|str|输入|待转换的DataType字符串形式，[AscendString](./cannkit-ascendstring-construction-and-destructor.md)类型。|

#### 返回值

输入合法时，返回转换后的DataType enum值，枚举定义请参考[DataType](./cannkit-ge-datatype.md)；输入不合法时，返回DT_UNDEFINED并打印报错日志。  

#### 约束说明

无  

#### 调用示例

```
ge::AscendString type_str("DT_UINT32");
auto data_type = AscendStringToDataType(type_str); // 8
```


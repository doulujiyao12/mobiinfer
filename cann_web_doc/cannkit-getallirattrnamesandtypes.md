# GetAllIrAttrNamesAndTypes

#### 函数功能

获取该算子所有的IR定义的属性名称和属性类型，包含普通和必选属性两种。  

#### 函数原型

```
graphStatus GetAllIrAttrNamesAndTypes(std::map<AscendString, AscendString> &attr_name_types) const;
```

#### 参数说明

|参数名|输入/输出|描述|
|:--------------|:----|:-----------------|
|attr_name_types|输出|所有的IR定义的属性名称和属性类型。|

#### 返回值

|类型|描述|
|:----------|:---------------------------------|
|graphStatus|GRAPH_FAILED：失败。 GRAPH_SUCCESS：成功。|

#### 异常处理

无  

#### 约束说明

无  

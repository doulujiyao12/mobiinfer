# SetOutputAttr

#### 函数功能

设置算子输出Tensor属性的属性值。

算子可以包括多个属性，初次设置值后，算子属性值的类型固定，算子属性值的类型包括：

* 整型：接受int64_t、uint32_t、int32_t类型的整型值。

  以int64_t为例：

  ```
  SetOutputAttr(const char_t *dst_name, const char_t *name, int64_t attr_value);
  SetOutputAttr(const int32_t index, const char_t *name, int64_t attr_value);
  ```

  设置属性值：

  ```
  GetOutputAttr(const int32_t index, const char_t *name, int64_t &attr_value) const;
  GetOutputAttr(const char_t *dst_name, const char_t *name, int64_t &attr_value) const;
  ```

  取值时，开发者需保证整型数据没有截断，同理针对int32_t和uint32_t混用时需要保证不被截断。
* 整型列表：接受std::vector\<int64_t\>、std::vector\<int32_t\>、std::vector\<uint32_t\>、std::initializer_list\<int64_t\>\&\&表示的整型列表数据。

* 浮点数：float32_t

* 浮点数列表：std::vector\<float32_t\>

* 字符串：string

* 布尔：bool

* 布尔列表：std::vector\<bool\>

#### 函数原型

```
Operator &SetOutputAttr(const int32_t index, const char_t *name, const char_t *attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const char_t *attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, const AscendString &attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const AscendString &attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, int64_t attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, int64_t attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, int32_t attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, int32_t attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, uint32_t attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, uint32_t attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, bool attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, bool attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, float32_t attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, float32_t attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<AscendString> &attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<AscendString> &attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<int64_t> &attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<int64_t> &attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<int32_t> &attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<int32_t> &attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<uint32_t> &attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<uint32_t> &attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<bool> &attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<bool> &attr_value);
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<float32_t> &attr_value);
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<float32_t> &attr_value);
```

#### 参数说明

|参数名|输入/输出|描述|
|:---------|:----|:----------------------------------|
|name|输入|属性名称。|
|index|输出|输出索引。|
|dst_name|输出|输出边名称。|
|attr_value|输入|需设置的int64_t表示的整型类型属性值。|
|attr_value|输入|需设置的int32_t表示的整型类型属性值。|
|attr_value|输入|需设置的uint32_t表示的整型类型属性值。|
|attr_value|输入|需设置的vector\<int64_t\>表示的整型列表类型属性值。|
|attr_value|输入|需设置的vector\<int32_t\>表示的整型列表类型属性值。|
|attr_value|输入|需设置的vector\<uint32_t\>表示的整型列表类型属性值。|
|attr_value|输入|需设置的浮点类型的属性值。|
|attr_value|输入|需设置的浮点列表类型的属性值。|
|attr_value|输入|需设置的布尔类型的属性值。|
|attr_value|输入|需设置的布尔列表类型的属性值。|
|attr_value|输入|需设置的字符串类型的属性值。|
|attr_value|输入|需设置的字符串列表类型的属性值。|

#### 返回值

|类型|描述|
|:---------|:----|
|Operator\&|对象本身。|

#### 异常处理

无  

#### 约束说明

无  

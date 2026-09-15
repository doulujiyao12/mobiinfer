# 构造函数

#### 函数功能

RuntimeAttrs类的构造函数。  

#### 函数原型

```
RuntimeAttrs() = delete;
RuntimeAttrs(const RuntimeAttrs &) = delete;
RuntimeAttrs(RuntimeAttrs &&) = delete;
RuntimeAttrs &operator=(const RuntimeAttrs &) = delete;
RuntimeAttrs &operator=(RuntimeAttrs &&) = delete;
```

#### 参数说明

无  

#### 返回值

无  

#### 约束说明

POD类型，当前不允许通过调用构造函数显式构造，可通过显式申请内存构造。  

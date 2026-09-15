# Attr

#### 函数功能

定义算子属性名称。  

#### 函数原型

```
OpAttrDef &Attr(const char *name);
```

#### 参数说明

|参数|输入/输出|说明|
|:---|:----|:------|
|name|输入|算子属性名称。|

#### 返回值

[OpAttrDef](./cannkit-opattrdef.md)算子属性定义。  

#### 约束说明

Attr属性名不能与python关键字及内置变量名相同，否则会导致未定义错误。

* python关键字：

  and, as, assert, break, class, continue, def, del, elif, else, except, finally, for, from, global, if, import, in, is, lambda, not, or, pass, raise, return, try, while, with, yield, False, None, True, nonlocal, await
* 内置变量名：

__inputs__, __outputs__, __attrs__, options, bisheng, bisheng_path, tikcpp_path, impl_mode, custom_compile_options, custom_all_compile_options, soc_version, soc_short, custom_compile_options_soc, custom_all_compile_options_soc, origin_func_name, ascendc_src_dir_ex, ascendc_src_dir, ascendc_src_file, src, op_type, code_channel, op_info, compile_op, get_code_channel, result, isinstance, attr, get_current_build_config, _build_args, get_dtype_fmt_options, shutil, os, get_kernel_source  

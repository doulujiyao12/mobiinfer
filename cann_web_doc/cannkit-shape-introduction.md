# 简介

Shape结构体用于描述一个tensor的shape，包含两个信息：

```
size_t dim_num_;
int64_t dims_[kMaxDimNum];
```

其中，dim_num_表示shape的维数，dims_数组表示tensor具体的shape。  

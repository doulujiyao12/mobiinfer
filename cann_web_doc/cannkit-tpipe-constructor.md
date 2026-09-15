# 构造函数

#### 功能说明

构造用来管理全局内存和同步的TPipe对象。  

#### 函数原型

```
__aicore__ inline TPipe()
```

#### 支持的型号

Kirin9020系列处理器

Kirin9030系列处理器

KirinX90系列处理器  

#### 注意事项

* 避免TPipe在对象内创建和初始化，TPipe在对象内创建时，可能会影响编译器对对象内常量的优化，引起scalar性能劣化。

* TPipe对象同一时刻全局只能存在一份，同时定义多个TPipe对象，会出现卡顿等随机行为。如果需要使用多个TPipe时，请先调用[Destroy](./cannkit-destroy.md)接口释放前一个TPipe。

#### 返回值

无  

#### 调用示例

```
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}
    __aicore__ inline void Init(..., TPipe* pipeIn)
    {
        // ...
        pipe = pipeIn;
        pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
        // ...
    }
private:
    // ...
    TPipe* pipe;
    // ...
};
extern "C" __global__ __aicore__ void example_kernel(...) {
    // ...
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    // ...
}
```


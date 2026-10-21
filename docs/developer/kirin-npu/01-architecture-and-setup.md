# 第一章：架构、使用方式与开发环境

[教程目录](README.md) · [下一章：在线编译](02-online-compilation.md)

## 1. 从应用到 NPU 经过哪些层

应用通常通过推理框架或运行时提交计算图，由厂商运行时与系统服务完成编译、内存准备和任务调度。应用开发者主要面对的是用户态 SDK/API，不是直接编写内核驱动或向硬件寄存器写指令。

![麒麟 NPU 软件栈：MNN 在线构图与 NNRT 离线模型路径通过运行时、系统服务和驱动访问 NPU](figures/01-software-stack.svg)

*图 1：以分层剖面展示两条主要接入路径。在线构图和离线模型导入在系统运行时汇合，再经驱动与固件访问 NPU；右侧标出接口边界。底部芯片为示意，不表示真实核数。*

在这条链路中，各层解决的问题不同：

| 层 | 主要职责 | 开发者通常修改什么 |
| --- | --- | --- |
| App | 文件管理、输入输出、请求取消、线程调度、界面状态 | 业务代码、Native 桥接和配置 |
| 推理框架 | 模型解析、形状推导、图优化、后端选择、CPU/NPU 分图 | 模型导出、后端与算子适配 |
| HiAI / CANN Kit | IR 表示、编译、执行接口与硬件相关优化 | 构图、编译选项、运行参数 |
| NNRT | 统一的设备选择、编译实例、执行器与张量接口 | C API 接入和资源管理 |
| 系统服务、驱动、固件 | 资源分配、任务下发、硬件执行 | 应用通常通过公开 API 使用 |

[CANN Kit 官方介绍](https://developer.huawei.com/consumer/cn/sdk/cann-kit/)列出了在线构图、模型量化、模型推理和单算子推理等能力。具体算子、数据类型及平台支持不能只由这个能力列表推断，需要查看所选版本的约束。

## 2. 有哪些使用路线

| 路线 | 开发者提供什么 | 编译在哪里发生 | 适合什么情况 |
| --- | --- | --- | --- |
| 框架在线编译 | MNN 等框架模型 | 手机运行时 | 希望保留框架生态，部署前不为每种芯片生成 OMC |
| 直接在线构图 | HiAI IR 或 NNRT 模型及参数 | 手机运行时 | 编写推理框架后端，或控制较小的自有图 |
| 离线图推理 | 匹配目标平台的 OMC 及相关文件 | 主机提前编译，设备仍需加载初始化 | 固定结构、固定输入档位、需要控制编译产物 |
| 单算子接入 | 受支持的算子描述和张量 | 按接口完成创建、准备和执行 | 将框架中的少数热点迁移到 NPU |
| AscendC 自定义算子 | 算子原型、Host Tiling、Kernel 与适配信息 | 开发主机及厂商工具链 | 现有算子不能表达或性能达不到目标 |

框架在线编译和直接构图可以调用不同的 API，但共同点是：设备获得计算描述后生成或恢复可执行模型。**“在线”不表示请求云端，也不要求联网。** “离线”指提前编译，同样在本机执行推理。

### 编译方式与计算划分

在线/离线描述**何时、在哪里编译**；全模型/混合部署描述**哪些计算交给 NPU**。两者需要分别设计：

| 计算划分 | NPU 执行范围 | 需要管理的资源与边界 |
| --- | --- | --- |
| 全模型 NPU | 模型的主要张量计算，包括可支持的编码器、embedding、decoder 和 head | 图、权重、激活、工作空间、I/O 与持久状态的完整工作集 |
| 混合部署 | 根据算子支持、资源预算和性能选择部分子图 | CPU/NPU 边界的拷贝、布局转换、同步与状态一致性 |

全模型 NPU 可以由一张或多张设备图组成，例如按 prefill/decode 分开编译。模型编译成功后，还需规划各图、权重和状态的驻留方式。内存不足时，可以调整分图、量化、上下文容量或常驻策略；混合部署也是一种工程折中。

单算子接口的支持集合与整图 IR 的支持集合不一定相同。不能因为 IR 有 MatMul，就认为任意版本的单算子 API 也能直接执行 MatMul。AscendC 也不是一个对所有麒麟设备都通用的 CUDA 替代层，需匹配平台工具链、运行能力与部署流程。

## 3. 先分清文件格式

| 文件 | 所处阶段 | 要点 |
| --- | --- | --- |
| HF `safetensors` / checkpoint | 原始或量化模型 | 不是设备可直接运行的图 |
| ONNX 与 ONNX external data | 导出与主机转换 | external data 是 ONNX 的大常量存储方式 |
| `.mnn` + `.mnn.weight` | MNN 运行模型 | MNN 可以将常量存储在单独文件中 |
| HiAI IR | 在线构图中间表示 | 描述算子、连边、属性和张量 |
| 在线生成并缓存的 `vision.om` | 设备编译缓存 | 与图、形状、精度及运行时环境有关 |
| OMG `--target=om` 产物 | 本项目工具链中的硬件无关模型表示 | 不能仅凭 `.om` 扩展名认为已完成目标芯片离线编译 |
| OMG `--target=omc` 产物 | 面向指定平台的离线模型 | 与 `--platform`、工具链及设备兼容性相关 |
| `SubGraph_0.weight` 等 | 厂商外置权重 | 必须与对应 OMC、目录和量化协议配套 |

最后两项的区分来自本项目使用的 OMG 参数定义和编译流程，见[官方 OMG 参数](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-overall-parameter)。不同代工具的命名可能变化，应保留实际命令与编译日志，不用改扩展名的方式“转换模型”。

三个“外置权重”格式也不能互换：ONNX external data、MNN `.weight`、OMC 外置权重分别由不同消费者解析。

## 4. 开发环境分成主机和设备两端

### 4.1 主机需要什么

| 工具 | 用途 | 什么时候需要 |
| --- | --- | --- |
| DevEco Studio 配套 HarmonyOS SDK | Harmony ARM64 工具链、NNRT 头文件和系统库链接文件 | 编译原生推理库与 App |
| CANN Kit 独立包 | 本项目 HiAI C++ 后端的头文件与 ARM64 库 | 编译 MNN HiAI 后端 |
| DDK-tools | OMG 模型转换、DOPT 量化、AscendC 配套工具 | 自行量化、离线编译或开发自定义算子 |
| 目标芯片插件 | 芯片平台配置及相关编译支持 | 生成对应芯片的 OMC / 自定义算子 |
| 主机 Python 与转换工具 | HF/ONNX/MNN 导出、量化和参考验证 | 构建自己的模型 |

工具下载入口：[CANN Kit / HiAI 下载](https://developer.huawei.com/consumer/cn/doc/hiai-Library/ddk-download-0000001053590180)、[CANN Kit 环境准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations)。选择相互匹配的工具包版本与平台插件。

本项目已有一组复现配置：CANN Kit `next-6.0.1.0`、DDK-tools 与 Kirin9030 插件 `next-6.1.1.0`。这是案例版本，不是要求所有项目长期固定这些版本，也不保证任意版本可以交叉混用。

### 4.2 `HARMONY_HOME` 路径配置

它是本仓库脚本的环境变量，指向 SDK Manager 安装的 **`openharmony` 目录**，不是 DevEco Studio 安装目录，也不是 DDK 或 CANN Kit 根目录。

```text
harmony-sdk/default/
├── openharmony/                         ← HARMONY_HOME
│   └── native/
│       ├── build/cmake/ohos.toolchain.cmake
│       └── sysroot/usr/include/neural_network_runtime/
└── hms/native/
    └── sysroot/usr/include/CANNKit/
```

`openharmony/native` 提供编译工具和 NNRT；`hms/native` 提供配套 CANN Kit 系统接口。独立 CANN Kit 的 `ddk/ai_ddk_lib` 又是另一组依赖，用于 MNN HiAI 后端。

在 Linux 上交叉编译，需要 Linux 版工具链；Windows 上已有 SDK 的路径不能直接拿给 Linux 编译器使用。当前仓库默认构建脚本使用平台 API 20；单个接口所需的最低版本仍需分别核对。

### 4.3 区分主机工具与设备运行库

OMG、DOPT 和主机编译器运行在开发主机。手机使用 ARM64 推理库和系统接口。把主机 `.so` 拷入 HAP 并不能提供 NPU 能力。

也不要混用 Android 和 Harmony 的 C++ 库。ABI、标准库命名空间及系统接口可能不同；本项目 Harmony 构建使用与之匹配的 HiAI 库，不直接套用 Android 动态符号桥。

## 5. 在本项目中准备环境

以下命令均在仓库根目录执行。先编辑 [sdk_config.sh](../../../project/harmony/sdk_config.sh)，设置工具包位置、`HARMONY_HOME` 和 App 源码目录。

```bash
# 准备 HiAI 依赖，编译 libMNN.so，并同步到本机 App 工程
bash project/harmony/setup.sh app
```

这条命令不负责生成模型，也不负责构建、安装 HAP。App 工程使用 [mobiinfra-oh / coevomind-app](https://github.com/doulujiyao12/mobiinfra-oh/tree/coevomind-app)，最终用 DevEco Studio 编译。App 自身的 CMake 生成 `libentry.so`，无需另外从本仓库手动复制这个文件。

如果只想分步执行：

```bash
bash project/harmony/01_prepare_hiai.sh
bash project/harmony/03_build_mnn.sh
bash project/harmony/04_sync_app.sh
```

只有自行构建离线模型时，才需要继续准备：

```bash
bash project/harmony/setup.sh offline-tools
```

`sdk_config.sh` 中的 `MOBI_*` 是本项目 Shell 环境变量，不是 NPU 硬件寄存器，也不是所有框架通用的编译宏。配置项含义就在文件注释中。

## 6. 先完成一个最小验证

环境验证宜从固定形状的小模型开始，依次检查：

1. 原生库能够编译、链接和被 App 加载。
2. 设备能够枚举目标加速后端，并获得运行时版本。
3. 一个小型、固定形状模型完成编译、加载、执行，并与 CPU 输出对照。
4. 同一个实例重复执行，不重新创建图。
5. 销毁、重新加载后再次执行，检查缓存与资源释放。
6. 扩展到目标模型的完整计算图，覆盖动态输入、持久状态和多步推理等实际需求。

版本号查询成功只证明调用到了相应接口。它不能单独证明一个完整模型的算子、内存和精度都受支持。

## 源码入口

- [Harmony 构建脚本](../../../project/harmony/build_64.sh)：当前编译选项。
- [SDK 准备逻辑](../../../project/harmony/sdk_setup.py)：依赖定位、解压、插件及 AscendC 初始化。
- [HiAI 后端构建](../../../source/backend/hiai/CMakeLists.txt)：库链接与公共头文件。
- [模型构建配置](../../../project/harmony/model_config.sh)：模型、校准数据、导出与验证输入。

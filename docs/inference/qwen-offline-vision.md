# Qwen3-VL-2B 与 Qwen3.5-2B：单图离线视觉

本方案在现有在线模型包中增加 Kirin9030 的单张视觉 OMC 和外置 FP16 权重。CPU LLM、EAGLE/MTP 和已有在线 MNN 视觉文件继续保留。它使用当前 App 的“离线 NPU”入口，不是全模型 NPU 方案。

## 执行方式和精度

| 部分 | Qwen3-VL-2B-Instruct | Qwen3.5-2B |
| --- | --- | --- |
| 离线 ViT | 原始视觉 checkpoint → FP16，单图含 pre、24 层 blocks 和全部 merger | 原始视觉 checkpoint → FP16，单图含 pre、24 层 blocks 和 merger |
| DeepStack | 保留视觉 5/11/17 层，独立投影后注入语言 0/1/2 层 | 没有 DeepStack，不构造零占位输出来模拟 |
| CPU LLM / draft | 原 MNN 官方 W4 主干和 EAGLE；embedding/head 沿用 W8 | 原 HQQ W4 主干和 MTP；embedding/head 沿用 W8 |
| 在线视觉 | 原 HQQ W8/group128 文件保留 | 原 FP16 MNN 文件保留 |
| 普通 KV / 递归状态 | 沿用原 CPU 配置 | 默认 `precision=high`，保持 FP32 |

Qwen3-VL 的新增离线视觉与原 W8 在线视觉采用不同权重精度，不能宣称两条视觉路径逐值相同。此次没有重新量化 LLM/draft，也没有引入 INT8 KV 或新的激活校准配置。

离线固定网格为宽 256 × 高 608，patch16、temporal2、merge2，每图 608 个 patch / 152 个视觉 token。图片预处理、分词、M-RoPE、采样和 CPU 缓存管理在主机端；完整视觉图在 NPU 执行。当前支持图片文件路径和多图；视频和内存图片入口会明确拒绝。

## 模型入口

- [Qwen3-VL-2B-Instruct-Eagle3-HiAI-MNN](https://modelscope.cn/models/fengerhu1/Qwen3-VL-2B-Instruct-Eagle3-HiAI-MNN)
- [Qwen3.5-2B-MTP-HiAI-MNN](https://modelscope.cn/models/fengerhu1/Qwen3.5-2B-MTP-HiAI-MNN)

下载完整目录，不能只下载新增的 OMC。App 模式与配置配合如下：

| 配置 | 用法 |
| --- | --- |
| `config.json` | App 选择在线 NPU 时走 MNN chunks；选择离线 NPU 时走单张 OMC，保留 EAGLE/MTP |
| `config_ar.json` | 与根入口相同，关闭 EAGLE/MTP |
| `config_online.json` / `config_online_ar.json` | 普通 MNN 在线配置，无离线 marker |
| `config_offline.json` / `config_offline_ar.json` | 明确要求完整视觉适配器；普通 `llm_demo` 缺少适配器时返回加载失败 |
| `config_cpu.json` / `config_cpu_ar.json` | 原 CPU 对照配置 |

新增的 `vision/config.json` 声明 `kirin_offline_npu2_vision_v2`、模型类型和 DeepStack 数量；图位于 `vision/om/vision_full.omc`，权重位于 `vision/weights/<hash>/SubGraph_0.weight`。根配置沿用 `kirin_offline_npu2_v1` marker，并通过 `vision_executor=omni_full_vision_v1` 选择新适配器。旧 Mobi marker 没有该字段，继续使用已发布的桥接实现。

## 导出与编译

使用 [prepare_qwen_offline_vision.py](../../project/harmony/prepare_qwen_offline_vision.py)，原始模型、现有 MNN 包和所有输出均放在 `/temp/models`。例如 Qwen3.5：

```bash
python project/harmony/prepare_qwen_offline_vision.py \
  --original-dir /temp/models/Qwen3.5-2B \
  --package-source /temp/models/Qwen3.5-2B-MTP-HiAI-MNN \
  --work-dir /temp/models/mobiinfer-work/qwen35_offline \
  --output /temp/models/Qwen3.5-2B-MTP-Hybrid-new \
  --ddk /path/to/DDK-tools-next-6.1.1.0 \
  --steps export,compile,package
```

Qwen3-VL 使用相同脚本，将原始目录换为 `Qwen3-VL-2B-Instruct`，现有 MNN 包换为 `Qwen3-VL-2B-Instruct-Eagle3-HiAI-MNN`，为 work/output 指定不同新目录。DDK 需要已安装 Kirin9030 插件；环境准备见 README。脚本直接读取视觉参数，不加载或导出 CPU LLM/draft，不执行新的 GPTQ 校准。

脚本首先检查结构和原始权重，将视觉模块映射为标准静态算子，再对照 HF 和 CPU ONNX，最后调用 DDK OMG。主要编译开关为 `--target=omc --platform=kirin9030 --save_weights_as_external_data=true --weight_data_type=FP16`。只有纯 NPU 分区和图 / 权重文件对均成功时才允许打包。`--steps compile` 可恢复编译；`--steps package` 可在已验证的 build 上组装新包。不要复用部分写入的导出目录重新执行 `export`。

## 引擎与 App

新适配器通过原有 `Llm::setNpuChunkExecutor` 入口注入完整视觉执行器。Omni 负责原有分词、M-RoPE、图片特征和 DeepStack 注入以及 CPU 生成；加载完整视觉时跳过 MNN 视觉 pre/blocks/post，不再额外加载 CPU ViT 权重。NNRT 后端沿用 Build → Construct → `HMS_HiAIExecutor_InitWeights` 流程，单个视觉 executor 跨请求保留。

**需要同时更新 `libMNN.so`、匹配公共头文件及 `offline_npu2` 源码，并重新编译 App。** 本次扩展了 C++ 插件接口，不能把新库与旧接口头编译的视觉插件混用。没有新增 NAPI 接口、HiAI 算子或 custom OPP，四个 HiAI 配套库与上一版相同。

本机 App 源码已同步；其他开发机器可合入模型仓库 `runtime/` 中配套 ZIP 的 `native/entry/` 文件，然后通过 DevEco Studio 编译 `libentry.so` 和 HAP。压缩包只含库、头文件、桥接源码和验证记录，没有模型权重，也没有预编译 HAP。

## 检查结果与边界

| 项目 | 结果 |
| --- | --- |
| 视觉映射和权重 | 两张真实图片与一张翻转图，逐路对照 HF；参数冻结为 FP16，参考运算仍为 CPU FP32 |
| DeepStack | Qwen3-VL 三路分别检查，顺序和独立 merger 保留；Qwen3.5 输出数为 1 |
| 厂商编译 | 两份模型均为 `NPU:1, CPU:0`；OMC 约 4 MB，外置权重分别为 814,294,528 / 663,173,632 字节 |
| 实际 CPU 桥接 | 单图与双图的 token IDs、M-RoPE 和注入 embedding 精确一致；与独立 CPU 调用的 prefill logits 最大绝对差为 0 |
| 生成和资源 | 两份模型的普通解码与 EAGLE/MTP 各执行单图/双图检查，覆盖文本、重复请求、预算、错误/取消后重试；executor 保留 |
| 兼容回归 | 旧在线模型的 CPU 视觉/LLM/draft 回归通过，原 Mobi 桥接保留 |
| 编译 | 新 `libMNN.so` 完成鸿蒙 ARM64 构建；App NAPI 和三个离线桥接源文件完成 ARM64 编译及 MNN 符号检查 |

**没有执行真机 OMC、没有构建 HAP，也没有连接手机。** CPU 回放是用已验证的视觉参考替代 NNRT 输出，证明宿主调度和 CPU 推理，不能证明设备上的 FP16 中间计算、内存或性能。离线权重仍占用设备内存，图权重分离不等于没有加载成本；本包不承诺 MTP/EAGLE 加速。换芯片或工具链须重新编译和验收。

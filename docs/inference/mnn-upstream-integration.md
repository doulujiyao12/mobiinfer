# MNN 主线合并与鸿蒙双模式验证（2026-10-18）

本次将 [MNN master dbf9ba61](https://github.com/alibaba/MNN/commit/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe) 合入 `integrate/mnn-master-20261018`。原开发版本为 `1fb0b954`，另保留 `backup/pre-mnn-20261018`；共同祖先为 `9d76f8a7`。保留双方提交历史，不用整目录覆盖或仅复制算子文件的方式升级。

本文记录主机可验证的构建、模型导出与数值边界。没有连接手机，不能据此承诺 HiAI/OMC 真机精度、速度或长期稳定性。

## 保留的执行方式

| 模式 | 视觉 | 语言与 KV |
| --- | --- | --- |
| 在线 NPU | 完整 MNN ViT：pre、六段 HiAI blocks、post；在设备上编译 IR/OM | MNN CPU |
| 离线 NPU，当前主方案 | ViT、merger、三路 DeepStack 合成一张 `vision_full.omc`，外置权重经 `HMS_HiAIExecutor_InitWeights` 加载 | MNN CPU；不加载 NPU decoder/head/embedding |
| Full NPU，备选实验 | 保留既有独立实现 | 保留既有独立实现，性能仍需优化 |

Mobi 是 Qwen3-VL。离线视觉复用已经冻结的 W4A16/group64 校准状态，固定 608×256，每图 152 个语言图像 token。语言及在线 MNN 视觉仍导入原用户 checkpoint 的 W8/group128 GPTQ 整数权重和 scale；文本 embedding、head 按原方案保留。**整包不是 W4；没有重新执行语言 GPTQ 校准，也没有把 AutoRound 的旧激活量化配置移植为 MNN 的激活计算。**

## 冲突处理与兼容修正

### HiAI 与鸿蒙工具链

- MNN 主线的 Android HiAI 动态符号桥使用 `__ndk1` ABI；CANN Kit 鸿蒙库使用 `__n1`。鸿蒙改为链接匹配的实际 HiAI 库，Android 保留主线动态加载方式。
- Android `native_handle`/AHardwareBuffer 代码受平台条件保护，鸿蒙不依赖 Android 头文件。
- 已验证的普通 HiAI 会话继续使用既有 IR 编译、OM 缓存、输入格式转换、输出 alias 及 Const 生命周期处理。保留主线显式插件会话路径与执行错误传播。
- 旧缓存目录升级为 `om_cache_v3`，避免升级后误用旧后端生成的缓存；首次在线运行需要重新编译缓存。
- OpenHarmony SDK Clang 15 无法编译主线新增的目标属性 FMLAL 路径。该编译器使用主线已有的 FP32 累加 NEON 兼容实现，支持的编译器仍保留 FMLAL 分派；未修改全局指令集要求。
- 目标库不携带构建主机 SDK 的绝对 RUNPATH。App 必须一起同步库与头文件，不能只替换一个 `.so`。

### Attention、元数据与精度

接受主线统一的 RuntimeManager 元数据传递，去掉本地重复的 `mMeta` 字段。语言会话保留 KV 元数据；视觉 processor/blocks 会话显式清空语言 KV 元数据。保留 Pillow 兼容的图像预处理、DeepStack 注入、M-RoPE 和擦除 prefix 后的位置恢复。

旧 CPU 视觉参考有一个可复现的问题：视觉 Attention 的 head_dim 为 64，但 processor 会话拿到了语言 head_dim=128 的 `attn_scale`。保存完全相同的 Q/K/V 后，独立 FP64 参考确认：

| 输出 | 相对 L2 误差 |
| --- | --- |
| 原 CPU 视觉 Attention 对正确 `1/sqrt(64)` 参考 | 0.45879276 |
| 原输出对错误的语言 `1/sqrt(128)` 参考 | 约 1.98e-7 |
| 合并修正后的输出对正确视觉参考 | 约 2.13e-7 |

因此不能强制新 CPU 视觉复现旧的错误输出。这是 CPU 视觉参考/回退路径的修正；不是已经测得 HiAI 真机精度提升。

主线将 Attention 缩放提前到 Q 打包，并新增 decode GQA 合批和更宽 KV 分块。浮点累加顺序的微小变化会被语言在线激活量化放大；本次实际观察到后续 logits 偏移及生成 token 分叉。保留浮点 KV 的 QK 后缩放；Mobi/Qwen3-VL 默认启用 `attention_preserve_rounding`，保持原 decode 分块和逐 head 计算顺序。其他模型仍可使用主线 decode 合批/宽块路径。关闭该兼容选项前必须重跑完整 decoding 数值对照，不能只验证 prefill。

对于已发布的 split MNN 视觉模型，默认保留原 `image_size` 预处理约定；主线按实际图片宽高计算的行为通过 `qwen_vl_use_image_dimensions` 显式选择，未拆分模型继续使用主线行为。离线 OMC 始终使用编译时固定网格。

### GPTQ 与新导出器

直接采用主线默认导出设置会发生三个兼容问题，本次均已通过实际 Mobi 导出与推理定位：

1. **投影融合跳过导入。** C4/FusedLinear 在 GPTQ 导入前合并 q/k/v、gate/up，原导入器只识别独立 Convolution，导致部分矩阵保留了重新取整的权重。指定语言或视觉 GPTQ 时关闭这组融合；导入器遇到未支持的 FusedLinear 直接失败。
2. **scale 类型与布局不匹配。** 主线默认 `scale_bit=16`，原导入器按 FP32 写入。FP16 的 `(zero,scale)` 与 FP32 的纯 scale 可能恰好占用相同字节数，单看文件大小及 payload 字节比较会漏检。语言 GPTQ 导出强制 `scale_bit=32`；导入器拒绝不兼容的 scaleStorage，审计按图中的 `readType` 检查布局。
3. **RoPE 计算路径改变。** 主线为 FP16 后端加入分段角度查表。它与旧直接 FP32 角度计算的舍入不同，首层实际输入中已观察到约 `1.6e-5` 差异；CPU W8 动态激活量化会放大该差异。Mobi/Qwen3-VL 的 CPU GPTQ 导出保留直接 FP32 M-RoPE，其他导出继续保留主线查表实现。不能将该 CPU 兼容选择直接作为其他 FP16 后端的长上下文方案。

这里的 FP32 指 scale，**不改变 W8 权重位宽**。非 GPTQ 导出保留主线默认融合及 scale 设置。冻结 W4 OMC 视觉导出使用自己的 DOPT 参数流程，不受此语言导入选项影响。

## 构建与模型导出

常规入口仍为 README 中的脚本。配置真实 OpenHarmony native SDK、CANN Kit、DDK 和 Kirin 插件后：

```bash
bash project/harmony/01_prepare_hiai.sh
bash project/harmony/03_build_mnn.sh
bash project/harmony/04_sync_app.sh
bash project/harmony/build_model.sh
```

`04_sync_app.sh` 同步 libMNN、四个 HiAI 库、MNN/LLM 公共头文件，以及 hybrid bridge 使用的 `llmconfig.hpp`/`ujson.hpp`。随后在 DevEco Studio 重编 App，生成与新库匹配的 `libentry.so` 和 HAP。

本次实际使用 OpenHarmony API20 native SDK、CANN Kit 6.0.1.0、DDK 6.1.1.0 和 Kirin9030 插件。SDK 原始归档 SHA256 为 `a315834ac133625efc912bd078f3e2b2550868d04aef1b5aa4f9679c8b3c9d8e`。主机模型验证使用现有 CANN Python 环境，并复用冻结 W4 校准 checkpoint；没有启动新的 GPU 量化。

离线模型编译实际运行 OMG，并生成 OMC/外置权重；通过既有编译日志和产物校验门槛。DDK 的 Python forkserver 需要本地进程通信权限，沙箱下初始化失败会进一步产生误导性的 unsupported-op 日志，应先修复工具执行权限再判断算子是否支持。

导出运行期间不要重链接正在使用的 MNNConvert 及其共享库。本次固定转换工具副本后完成模型导出，避免并行构建替换 `.so` 造成无效 ELF。

## 回归方式与验收边界

- 实际构建 Linux CPU 引擎、MNNConvert，以及鸿蒙 ARM64 HiAI `libMNN.so`、LLM/demo/test 工具。
- CPU 算子测试、Attention 测试，以及 NNRT 加载/失败处理、增量 KV、协议测试。NNRT 主机测试使用调用替身，不是设备驱动执行。
- 对源 GPTQ 独立解包，审计 196 个语言矩阵及 144 个 MNN 视觉矩阵的整数编码、FP32 scale 布局、零点与分组；不通过则拒绝发布包。
- 对单图/双图检查与 HF 一致的 token、预处理、DeepStack、M-RoPE；用独立 Python 注入的视觉特征验证生产 C++ 桥接。
- 旧引擎与新引擎比较首步及连续 16 步 decoding 的真实 logits 和生成 token。`project/harmony/compare_snapshots.py` 对整数要求完全一致，对浮点使用 `atol=1e-4, rtol=1e-5`，并记录是否逐字节相同。
- 使用独立图片评估冻结 W4 视觉，检查最终图像 embedding 和每路 DeepStack。该图片未参与冻结校准；它不能代表完整视觉任务质量。

示例：先分别用旧、新 CPU host_driver 执行相同模型包和图片，再比较快照：

```bash
python transformers/llm/export/kirin_hybrid_offline/verify_host.py \
  --driver /path/to/host_driver --package /path/to/package \
  --build /path/to/vision_build --model /path/to/original_mobi \
  --image /path/to/heldout.png --output /path/to/snapshots \
  --copies 1 --decode-steps 16
python project/harmony/compare_snapshots.py \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --output /path/to/comparison.json
```

上述 `host_driver` 用 CPU ONNX 视觉输出替代设备调用，执行真实 MNN CPU LLM。`verify_host.py` 的桥接自洽检查和跨版本快照对照必须分别执行：两条路径都使用同一份错误语言图时，桥接自洽检查仍可能通过。

本次结果见 [验收证据](mnn-upstream-integration-evidence.json)：304 项 CPU 算子测试通过，其中包含 13 项 Attention 测试；32 项构建/模型脚本测试、5 项 GPTQ 导出兼容测试、9 项 NNRT 调用替身测试和 1 项桥接协议测试通过。

新导出模型的单图/双图分别为 167/321 tokens；prefill 和连续 16 步 decoding 的 logits、生成 token、桥接输入均与合并前逐字节一致。失败重试、纯文本请求及 KV prefix 复用的完整测试输出也一致。在线 MNN 模型在修正视觉 scale 的同一 CPU 运行时上比较，新旧模型的单图/双图 prefill logits 与前三个生成 token 同样逐字节一致。

整份 CPU LLM 权重文件 SHA256 保持为 `23c072fb9a870812cdf4c859bf2fe57b3b9594852fb8794dc34dbf62c0c29095`；冻结 W4 视觉外置权重也与已发布版本逐字节一致。重新编译得到的视觉 OMC 为 5,265,008 bytes，外置权重为 251,344,384 bytes。

本机交付目录为 `.artifacts/mnn-master-20261018/`：

- `native/`：真实鸿蒙 ARM64 libMNN、HiAI 库、配套头文件和命令行工具。
- `model/config.json`：离线单图入口；`model/config_online.json`：在线 MNN 入口。
- `integration-report.json` 和 `validation/`：验收汇总与测试记录；模型包内也保留精度报告和文件哈希清单。

大文件交付目录由 Git 忽略；构建中间产物保留于 `/tmp/mobi-mnn-validation`。失败候选不作为交付模型。

公开 OpenHarmony SDK 不含 App 使用的 HMS `CANNKit/hiai_helper.h`、`hiai_options.h` 和 `libhiai_foundation.so`。本机可验证 App 对新 MNN 的原生接口、ARM64 编译和符号依赖，但完整 libentry/HAP 仍需具有这些组件的 Huawei HMS native SDK。未使用测试替身伪造该 SDK，也未连接手机。未测试 Windows DevEco/BiSheng、QNN/其他硬件后端及真机在线编译/OMC 执行；性能问题仍然需要后续设备验证与优化。

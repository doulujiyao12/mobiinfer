# MNN 主线合并与鸿蒙双模式验证（2026-10-18）

本次将 [MNN master dbf9ba61](https://github.com/alibaba/MNN/commit/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe) 合入 `integrate/mnn-master-20261018`。原开发版本为 `1fb0b954`，另保留 `backup/pre-mnn-20261018`；共同祖先为 `9d76f8a7`。保留双方提交历史，不用整目录覆盖或仅复制算子文件的方式升级。

本文记录主机可验证的构建、模型导出与数值边界。后续用户已反馈在线 HiAI 注册及缓存修复可用；本文的主机回归结果仍不能代替 HiAI/OMC 真机精度、速度或长期稳定性验收。

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
- 修复单库 HiAI 后端注册遗漏：CMake 为 `MNN_NPU` 设置 `MNN_NPU_INTEGRATED_REGISTRATION` 后会关闭原静态注册，`Backend.cpp::registerBackend()` 必须在 `MNN_NPU_ENABLED` 下调用 `registerHiAIRuntimeCreator()`。缺少该调用时，即使库包含 HiAI 代码也会回退 CPU。实际手机日志已观察到 `requested=hiai actual=0 hiai=8 enabled=1`，六个视觉 chunk 均回退 CPU，因此没有 OM 编译或缓存；本次视觉耗时约 11.41 秒。以实际 `Backend.cpp` 和替代 registrar 的并发主机验证中，修复前注册次数为 0、修复后恰为 1；鸿蒙 ARM64 库已重新构建。用户随后反馈验证可用；注册链主机测试本身不执行 HiAI 驱动。
- 已验证的普通 HiAI 会话继续使用既有 IR 编译、OM 缓存、输入格式转换、输出 alias 及 Const 生命周期处理。保留主线显式插件会话路径与执行错误传播。
- 旧缓存目录升级为 `om_cache_v3`，避免升级后误用旧后端生成的缓存；首次在线运行需要重新编译缓存。
- chunk OM 缓存现已默认开启；仍可通过 `MNN_HIAI_CACHE_OM_BY_CHUNK=OFF` 显式关闭。缓存位于 `<模型目录或 npu_model_dir>/chunk_i/om_cache_v3/shape_*/vision.om`，不是旧的 `om_cache_v2`。模块克隆继承源会话的 chunk 目录，避免新后端恢复为 `.` 后静默跳过缓存。成功写入及复用分别打印 `[NPU_CACHE] saved`、`hit`/`validated`；关闭开关、缺失目录及写入失败也会记录原因。OM 通过独立临时文件写完后原子替换，避免并发写入共用一个临时文件。
- 后续代码复查发现，原 shape key 未包含图内容、外置权重、精度和 HiAI 版本，同目录替换模型时可能误用旧 OM。现在 HiAI 模块加载时计算实际图及外置权重内容标识，缓存键同时包含精度和 HiAI 版本，Session 克隆保留源标识。读取外置权重失败时禁用该模块的磁盘 OM 缓存；不按文件时间戳推断内容相同，因为快速同尺寸覆盖可保留相同时间戳。内容计算增加模块加载时的读取开销，不在逐 token 推理中执行。键变化后每个 shape 首次使用需重新编图一次；无需重新导出模型。
- 磁盘命中与图已驻留分别验收。后续手机日志确认：六个 chunk 全部 `hit`/`validated` 时，重新加载后的首图仍需 7.541/6.894 秒，同实例下一张为 1.220/1.236 秒；最近一次六次 HiAI Process 调用合计约 0.647 秒。原命中点位于所有算子 `onResize` 之后，因此仍有量化权重还原、主机端 IR 对象构造及 OM 加载开销。现增加 Pipeline 在算子 resize 前加载图的入口，仅对 HiAI 主后端非 Copy 命令启用；张量分配、输入输出拷贝及正常执行保留。正常编译或旧缓存加载成功后保存 `om_cache_v3/fast_*.meta`，按模型标识、精度、HiAI 版本和完整命令张量形状/类型/格式区分，记录实际 DDK 槽位及 DeepStack 输出别名，附内容校验。新入口先校验元数据、OM 大小及驱动返回的 I/O 大小，通过后才跳过算子 resize；缺失、损坏或不兼容时回原流程。已有 OM 第一次使用新库时需要补齐索引，之后重新加载才可走快速路径。
- 快速路径通过私有 mmap 提供 OM 数据，减少整份 OM 的堆分配/拷贝；仍使用相同的 HiAI 同步 Load、布局转换与 Process。成功日志包含 `fast=1 load_ms=... io_ms=... skipped_op_resize=...`，用于区分设备加载与执行成本。真实 SDK ARM64 构建、主机 Pipeline 跳过 resize/保留 Copy 验证、实际后端方法的调用替身测试（同尺寸输入顺序、DeepStack 别名、损坏/不兼容/模型变化回退）及 CPU 数值回归已通过。后续手机日志确认六个 chunk 均命中 `fast=1`，首图 3.762 秒、同实例第二张 1.153 秒；首图缓存加载合计 2.197 秒，两次 HiAI Process 分别约 0.656/0.650 秒。磁盘 OM 仍需加载到设备，不能将同实例约 1.2 秒的热执行时间作为冷加载保证。
- 进一步增加模块级 OM 入口：普通路径成功执行后保存 `om_cache_v3/module_cache_v1.meta`，包含公开输入输出、准确的 DDK 槽位、布局和 DeepStack 别名，以及已校验的源模型/外置权重/OM 文件代次。后续同形状重载在打开原始 `.mnn` 之前检查元数据，直接加载已有 OM，跳过该 ViT blocks 模块的原始权重读取、常量准备和内容指纹计算；CPU LLM 与视觉 pre/post 仍正常加载。原始文件继续保留供失效回退使用。校验同时覆盖设备精度及 HiAI 版本、inode/大小、纳秒 mtime/ctime；文件写入刻度尚未过去、平台不支持可靠文件代次、内容参与形状计算、存在 CPU 计算或接口不能完整映射时均保留原路径。新增索引前后复核源文件代次；损坏、形状变化及驱动拒绝也回退，不把一次 NPU 执行错误隐藏为 CPU 结果。旧 OM 无需清理或重新导出，但需要一次正常运行补齐索引。可用 `module_saved` 和 `module_hit ... source_weights_loaded=0` 区分索引生成与原始权重跳读。主机已覆盖真实六段 Mobi 接口和 DeepStack 数量、源文件零打开、输入/输出重排、缓存失效及回退；真实后端方法以驱动替身验证，尚未取得该模块入口的真机精度和速度结果，不能保证冷加载达到 1.1 秒。
- 有效模块索引对应的 OM 在 `Module::load` 返回前完成 HiAI 加载及设备输入输出缓冲区初始化，日志为 `module_hit ... phase=model_load source_weights_loaded=0 load_ms=...`。后续同形状首图直接执行，加载时间计入模型加载阶段；这是准备工作前移，不代表设备加载本身变快。缓存删除或驱动拒绝时，在模型加载阶段准备原始 MNN 模块。首次编译、没有完整索引的旧缓存或输入形状变化仍可能在收到图片后加载/编译，不能预先准备未知形状，也不能保证首次推理达到同实例热执行时间。
- 模块缓存保留原始 API 的模式约束：固定形状模块明确拒绝分离 forward 模式，Resize tracing 交回原始 Session；普通推理继续直接使用已加载 OM。设备导入在维度和 I/O 缓冲区全部验证成功后才提交，失败时显式卸载并清理临时状态，重复加载不覆盖有效实例。执行前检查输入输出缓冲区，拷贝失败及设备失败均保持错误返回，并完成执行阶段收尾。 克隆及延迟回退前后重新检查源文件代次；已加载期间源文件被替换时要求重新加载，避免旧 OM 与新源模型混用。
- App 的在线模式选择日志不证明 HiAI 已执行或缓存已保存。鸿蒙缓存诊断同时输出到 HiLog INFO 和 stdout，供 App Native 日志捕获；`visual_runtime` 记录实际后端（`hiai=8`、`actual=0` 表示 CPU 回退），`vision_begin` 记录视觉处理入口。App 状态解析识别 chunk CPU 回退以及缓存保存/校验成功，后两者只计为图就绪，不能计为 NPU 执行成功。纯文字请求不执行动态 ViT 编图。chunk OM 扫描不限定缓存版本，并覆盖旧 `vision.om.tmp` 和并发写入使用的 `vision.om.tmp-*`；临时文件及零字节 OM 单独标为未完成，也可由缓存清理入口删除。
- App 在真实进入在线编译时显示“首次加载，在线编译中，请等待”，通过独立 Native 编译事件计数驱动，Debug 日志关闭时仍可提示；缓存命中不触发，多个 chunk 提示节流，页面离开时停止轮询。提示不进入 token 流或模型回答。
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

缓存修复后的再次复查覆盖 `mobiinfer` 当前未提交变更，以及 App 的 `coevomind-app-dev` 相对 `coevomind-app` 的四个提交。主机及鸿蒙 ARM64 库重新构建成功，App 的 `napi_init.cpp` 通过真实 SDK ARM64 编译；54 项表达式测试、32 项脚本测试、9 项 NNRT 调用替身测试及 1 项桥接协议测试通过。缓存内容标识、实际 Module 加载与 Session 克隆、缓存键失效、App 状态解析及缓存扫描/删除另有临时回归验证，未使用 NPU 驱动替身的结果声称设备执行成功。单图完整 MNN 视觉链路的 prefill logits/前三个 token，以及离线参考视觉注入的 prefill/连续 16 步 decoding logits 和 token，均与既有快照逐字节一致。新库和 29 个公共/桥接头文件已同步并比对到 App 源码；完整 HAP 及本次复查新增修正仍需 DevEco/真机验证。复查记录保存在本机 `.artifacts/review-20261018/`，未重新量化或导出模型。

新导出模型的单图/双图分别为 167/321 tokens；prefill 和连续 16 步 decoding 的 logits、生成 token、桥接输入均与合并前逐字节一致。失败重试、纯文本请求及 KV prefix 复用的完整测试输出也一致。在线 MNN 模型在修正视觉 scale 的同一 CPU 运行时上比较，新旧模型的单图/双图 prefill logits 与前三个生成 token 同样逐字节一致。

整份 CPU LLM 权重文件 SHA256 保持为 `23c072fb9a870812cdf4c859bf2fe57b3b9594852fb8794dc34dbf62c0c29095`；冻结 W4 视觉外置权重也与已发布版本逐字节一致。重新编译得到的视觉 OMC 为 5,265,008 bytes，外置权重为 251,344,384 bytes。

本机交付目录为 `.artifacts/mnn-master-20261018/`：

- `native/`：真实鸿蒙 ARM64 libMNN、HiAI 库、配套头文件和命令行工具。
- `model/config.json`：离线单图入口；`model/config_online.json`：在线 MNN 入口。
- `integration-report.json` 和 `validation/`：验收汇总与测试记录；模型包内也保留精度报告和文件哈希清单。

大文件交付目录由 Git 忽略；构建中间产物保留于 `/tmp/mobi-mnn-validation`。失败候选不作为交付模型。

公开 OpenHarmony SDK 不含 App 使用的 HMS `CANNKit/hiai_helper.h`、`hiai_options.h` 和 `libhiai_foundation.so`。本机可验证 App 对新 MNN 的原生接口、ARM64 编译和符号依赖，但完整 libentry/HAP 仍需具有这些组件的 Huawei HMS native SDK。未使用测试替身伪造该 SDK，也未连接手机。未测试 Windows DevEco/BiSheng、QNN/其他硬件后端及真机在线编译/OMC 执行；性能问题仍然需要后续设备验证与优化。

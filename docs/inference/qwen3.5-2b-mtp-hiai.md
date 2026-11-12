# Qwen3.5-2B：在线 HiAI 视觉与 CPU 原生 MTP

本文对应 Qwen 官方 **Qwen3.5-2B dense** checkpoint。视觉 Transformer blocks 在线编译到麒麟 HiAI NPU，视觉 pre/post、语言模型、原生 MTP、缓存维护和采样在 CPU 执行。原来的 Qwen3-VL 普通解码与 EAGLE-3 配置继续保留。

本次完成了主机 CPU 数值检查、真实 MNN 模型推理和鸿蒙 ARM64 交叉编译。**尚未验证手机上的 HiAI 编译、执行、缓存、内存峰值和速度；不能将主机结果视为真机验收。**

2026-11-12 增加了 [Kirin9030 单图离线视觉](qwen-offline-vision.md)，完整 ViT 编译为一张 OMC，FP16 权重外置，CPU MTP 继续保留。原在线和 CPU 模式继续支持。

## 1. 架构与执行划分

| 部分 | Qwen3.5-2B 的结构 | 本次处理 |
| --- | --- | --- |
| 视觉编码器 | 24 层，hidden 1024，patch 16，merge 2 | 分为 6 个 MNN blocks 图，由 HiAI 在线编译；pre/post 在 CPU |
| DeepStack | 该 checkpoint 不包含 DeepStack 输出 | 显式使用 `has_deepstack=false`，不沿用 Qwen3-VL 的三路特征注入 |
| 语言主干 | hidden 2048，共 24 层 | 全部在 MNN CPU 执行 |
| 线性注意力 | 18 层 Gated DeltaNet，带卷积窗口与递归状态 | 使用 CPU `LinearAttention`，投机验证后仅提交已接受的前缀 |
| 全注意力 | 每 4 层中的最后一层，共 6 层 | 普通 CPU Attention KV Cache |
| 位置编码 | partial RoPE + 三轴 interleaved mRoPE | 由当前图文输入的位置生成接口提供，不以 KV 长度代替图像位置 |
| 原生 MTP | checkpoint 中的 `mtp.*`，一个全注意力预测层 | 单独导出 `mtp.mnn`，CPU 生成候选，主模型验证 |

Qwen3.5 的 MTP 与此前 Qwen3-VL 使用的 EAGLE-3 是不同的模型分支。这里使用原始 checkpoint 自带的 MTP 参数，不复用 Qwen3-VL 的 EAGLE 权重。配置和参数布局可查阅 [Qwen 官方配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)及 [vLLM 的原生 MTP 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)。

## 2. 量化与缓存精度

| 对象 | 当前配置 |
| --- | --- |
| LLM 主干 Linear | HQQ W4，group 64 |
| MTP 主干 Linear | HQQ W4，group 64 |
| 词嵌入 / LM head | W8，group 64；主模型词嵌入复用其 head 权重 |
| 量化 scale 存储 | FP16 |
| 视觉权重 | FP16 |
| CPU 动态激活量化 | `dynamic_option=2`，按权重分组进行在线输入量化 |
| CPU KV 与递归状态 | 默认 `precision=high`，保留 FP32 |

HQQ 是权重量化，本流程没有使用图片观察激活，也没有重新执行 GPTQ 校准。CPU 的在线激活量化由实际推理输入计算 scale；它与导出时的权重量化是两件事。归一化、RoPE 等也不是全部转换成 INT8。

视觉 W8 候选的最终特征相对 L2 误差约为 **5.58%**，没有通过本次预先设置的 5% 门槛。改用 FP16 后降至 **0.286%**，余弦相似度约 **0.999996**。最终选择 FP16 视觉以优先保留精度，语言与 MTP 主干仍为 W4。

默认 CPU 高精度还考虑到 Qwen3.5 的递归状态精度。现有 ARM 低精度路径会使用 FP16 状态，本次没有真机验证它在长序列上的误差，因此没有将其作为新模型默认配置。降低 CPU 精度不会改变磁盘上的 W4 权重位宽，但会改变运行时数值路径。

## 3. 下载与导出

模型及所有模型中间产物必须位于 `/temp/models/`。本机目录登记在 [models.md](../../models.md)。主机转换器与鸿蒙库仍按仓库 [README 的编译说明](../../README.md#编译与集成)准备。

在支持 `qwen3_5` 的模型导出 Python 环境中执行：

```bash
python project/harmony/prepare_qwen3_5_mtp.py \
  --original-dir /temp/models/Qwen3.5-2B \
  --export-dir /temp/models/mobiinfer-work/qwen35-mtp/export \
  --mnnconvert /absolute/path/to/host/MNNConvert \
  --output /temp/models/Qwen3.5-2B-MTP-HiAI-MNN-new
```

脚本复用本地完整 checkpoint；缺失时通过 ModelScope SDK 下载 `Qwen/Qwen3.5-2B` 到指定目录。未完成且不受下载标记管理的本地目录会报错，避免覆盖其他权重。导出目录和输出目录应为空；`--reuse-export` 仅用于打包已经完成的同精度导出。

脚本依次检查模型结构、对照真实图片的视觉映射、导出视觉、导出主模型与 MTP、导出 tokenizer、生成配置及文件哈希。默认视觉 FP16；`--visual-quant-bit 8` 可生成 W8 实验候选，应重新进行精度验收。脚本中的视觉参考检查用于验证映射，不是激活校准。

已发布运行包：[fengerhu1/Qwen3.5-2B-MTP-HiAI-MNN](https://modelscope.cn/models/fengerhu1/Qwen3.5-2B-MTP-HiAI-MNN)。下载完整目录：

```bash
modelscope download --model fengerhu1/Qwen3.5-2B-MTP-HiAI-MNN \
  --local_dir /temp/models/Qwen3.5-2B-MTP-HiAI-MNN
```

最新发布包包括在线 / 离线 / CPU 配置、中文说明、验证报告及 `runtime/` 下的配套鸿蒙库压缩包。增加单图离线视觉后共 49 个发布文件，远端大小和 SHA256 已逐项核对；真机验收状态仍为未完成。最初仅在线版本的 37 文件记录保留作历史对照。

当前完整运行包：

```text
/temp/models/Qwen3.5-2B-MTP-HiAI-MNN/
├── config.json              # 在线 HiAI 视觉 + CPU 主模型 / MTP
├── config_ar.json           # 在线 HiAI 视觉 + CPU 普通解码
├── config_cpu.json          # CPU 视觉 + CPU MTP，供主机验证
├── config_cpu_ar.json       # 全 CPU 普通解码对照
├── llm.mnn / llm.mnn.weight
├── mtp.mnn / mtp.mnn.weight
├── visual_pre.mnn / visual_pre.mnn.weight
├── visual_blocks_npu_0..5.mnn 及配套 .weight
├── visual_post.mnn / visual_post.mnn.weight
├── tokenizer.mtok
├── llm_config.json
└── manifest.json
```

上述 **MNN 格式**文件用于在线 NPU / CPU。新增离线资产保存在 `vision/`，通过 `HMS_HiAIExecutor_InitWeights` 加载外置视觉权重，另有 `config_online*.json` 和 `config_offline*.json` 独立入口，详见 [单图离线视觉](qwen-offline-vision.md)。MNN 图与 `.weight` 文件需要配套保留。MTP 当前单独存放其 head 权重；共享 embedding/head 不代表主模型与 MTP 已实现设备内存去重。

## 4. MTP 的启用和执行

关键配置如下，四个入口使用同一套模型权重：

```json
{
  "backend_type": "cpu",
  "precision": "high",
  "dynamic_option": 2,
  "speculative_type": "mtp",
  "mtp_type": "qwen3_5",
  "mtp_model": "mtp.mnn",
  "hidden_states": true,
  "draft_predict_length": 2,
  "sampler_type": "greedy",
  "reuse_kv": false,
  "chunk": 0,
  "attention_mode": 8,
  "visual_blocks_backend_type": "hiai"
}
```

由配置选择生成策略，不扫描目录来自动启用 MTP。`config_ar.json` 的 `speculative_type` 为空；旧 Qwen3-VL 的 `speculative_type=eagle` 仍使用 EAGLE，普通模型仍使用自回归。

MTP 将归一化后的下一 token embedding 和前一位置 hidden 按顺序拼接，经投影与预测层得到候选。主模型用因果批量前向验证候选。普通 KV 丢弃未接受的后缀；线性注意力保存候选输入及已提交状态，在下一次前向前重放已接受的前缀，避免把被拒绝 token 写进卷积窗口和递归状态。

需要将整次生成预算交给 `response`，例如 `response(prompt, &stream, "", 128)`。反复调用 `generate(1)` 会走普通解码兼容路径，不能用于证明 MTP 加速。

## 5. 鸿蒙库和 App

新模型需要本次更新后的 `libMNN.so`。源代码仍通过仓库的 Harmony 构建脚本编译；本机已生成的库与头文件位于：

```text
outputs/qwen3.5-2b-mtp-hiai/native/
```

最新配套 ZIP 的 `native/entry/` 对应 App 仓库的 `entry/` 布局，含 `libMNN.so`、配套 HiAI 库、头文件和单图离线桥接源码；将它们一起集成到当前 App 源码后用 DevEco Studio 重新构建。没有新增 NAPI 接口，App 编译时正常生成 `libentry.so`。

**新增离线模式需要同时更新 `libMNN.so`、匹配头文件和 `offline_npu2` 桥接源码**，不能仅替换库。四个配套 `libhiai*.so` 与上一版相同，可继续复用；没有新增 HiAI 算子或自定义算子包。

配套 [鸿蒙 ARM64 库包](https://modelscope.cn/models/fengerhu1/Qwen3.5-2B-MTP-HiAI-MNN/resolve/master/runtime/mnn-ohos-arm64-qwen35-mtp-hiai.zip) 已随模型上传。压缩包内仍以 `native/entry/` 为集成入口，不包含模型权重。

App 选择**在线 NPU**或 **离线 NPU**，加载更新后的根 `config.json`；普通解码对照使用 `config_ar.json`。两种模式都保留 CPU MTP；独立离线入口为 `config_offline.json`，其 OMC 面向 Kirin9030。

## 6. 已执行的检查

验证环境为 Linux x86_64 CPU，原始 HF 模型使用 FP32/eager，MNN CPU 使用高精度、贪心解码。原模型与导出映射使用同一 checkpoint；模型文件均核对 SHA256。

| 检查 | 结果与边界 |
| --- | --- |
| 原始 HF 图文推理 | 数字问答、杭州介绍、猫图识别正常 |
| 主模型 Python 映射 | prefill + 8 步 decode；线性注意力层、全注意力层、末层和 logits 对照，相对误差约 `10⁻⁶` |
| 原生 MTP Python 映射 | 独立 HF 预测层对照，prefill + 2 步 decode，约 `2×10⁻⁷` |
| 实际 MNN tokenizer | 3 个样例的输入 token IDs 与 HF 完全一致；图像样例为 256 × 608，175 个总输入 token |
| 实际 MNN FP16 ViT | 真实图片最终特征相对 L2 约 0.286%，余弦约 0.999996；不包含 DeepStack |
| 实际 MNN W4 模型 | MTP / 普通解码均完成图文问答；预算 1/2/5/16、EOS 后重复请求、重置及逐 token 续接通过 |
| draft 执行证据 | 算子回调观察到 MTP 专有投影 `/mtp/fc/Linear` 实际运行 |
| 状态回滚 | 全接受、部分接受、全拒绝、连续验证、flush 与非法回滚检查通过 |
| 旧模型回归 | Qwen3-VL-2B 普通解码与 EAGLE-3 图文、预算、EOS、重置和续接检查通过 |
| CPU 算子回归 | Attention 13 项、LinearAttention 13 项测试入口通过；其中依赖 Apple 文件接口的两项明确跳过 |
| 鸿蒙构建 | API 20 AArch64 ELF 与 HiAI 依赖检查通过；未连接手机 |

检查中修正了三处实际问题：

1. **FP32 参数在加载时丢失精度**：checkpoint 混合 BF16 矩阵与 FP32 归一化参数，先整体加载为 BF16 会破坏后者。Qwen3.5 导出改为 FP32 加载，再按目标格式量化。
2. **重复请求 decode 位置偏移**：tokenize 在请求 reset 前执行，旧代码把上一轮长度计入新 decode。按是否复用 KV 选择前缀偏移后，同一请求连续 5 次的输出及跟踪到的注意力结果完全一致。
3. **首行 logits 采样范围**：偏移为 0 时也必须按指定 vocabulary 长度截取，不能忽略批量验证首行的切片。

## 7. 精度与性能的实际限制

**没有建立 W4 与原始浮点模型无损对齐的结论。** 将实际 `.weight` 中的整数与 scale 独立反量化，再在 PyTorch 计算，两条文字样例的 logits 相对原模型仍有约 28.9% 和 35.4% 的 L2 差异。MNN 相对这份冻结反量化参考的差异约为 2.1% 和 3.1%。这区分了存储权重量化与运行时计算的误差；少量样例能回答不代表完整任务质量已经达标。

`dynamic_option=2` 将同输入单 token / 批量验证的一个样例 logits 差异从约 7.84% 降至 1.81%，但仍可能改变接近的候选排序。杭州介绍的 48-token 对照前 47 个 token 一致，另一些提示更早分叉；**不保证 MTP 与普通解码逐 token 相同**。

两种解码都能回答“2”、识别猫，并生成相关的中文介绍。本次不是标准图文评测，也没有新做 GPTQ 校准；后续可用冻结的独立评测集比较 GPTQ 等候选与当前 HQQ 包。

**MTP 不保证更快。** 当前主机样例中，长回答启用 MTP 比普通解码更慢；draft、额外 head、验证及状态提交的开销需要足够高的接受率才能抵消。该结果不是手机跑分，手机端还需要分别测量视觉在线编译/缓存加载、NPU 图执行、CPU prefill 与 decode。

本次明确保留的支持边界：

- 已验收的是 Qwen3.5-2B dense、单个原生 MTP 层、共享词嵌入；MoE、多 MTP 层或独立 MTP embedding 会明确拒绝，不能直接套用本包。
- 原生 MTP 当前要求贪心采样，draft 长度限制为 1～8，默认 2；未实现随机采样的拒绝采样校正。
- 逐 token 续接、缺少完整 target hidden 前缀的分块 prefill、保留旧前缀的部分续接会回到普通解码；不会用缺失状态继续生成 draft。
- 主机验证覆盖单图问答，未验收视频、多图、工具调用、超长上下文或手机 FP16 递归状态精度。
- HiAI 不支持的图或设备资源不足仍可能导致在线编译失败。主机转换成功和 ARM64 编译成功都不能证明设备驱动接受该图。

可复用的集成检查程序为 [validate_qwen3_5.cpp](../../project/harmony/tests/validate_qwen3_5.cpp)、[validate_split_vision.cpp](../../project/harmony/tests/validate_split_vision.cpp) 和原来的 [validate_eagle.cpp](../../project/harmony/tests/validate_eagle.cpp)。本轮机器可读结果随原生库保存在 `outputs/qwen3.5-2b-mtp-hiai/validation/`，模型哈希见运行包 `manifest.json`。

# Qwen3-VL-2B + EAGLE-3：ViT 在线 HiAI，语言推理使用 CPU

本方案使用 MNN 官方发布的 Qwen3-VL-2B-Instruct 主模型和配套 EAGLE-3 draft。视觉 transformer blocks 使用本仓库的 HiAI 后端，在鸿蒙手机上在线编译；语言主模型、draft、视觉前后处理在 CPU 上运行。

EAGLE-3 是另外训练的 draft，并非 Qwen3-VL-2B 原始权重中自带的 MTP 模块。本方案也没有把 Mobi 的量化权重拼入 Qwen 模型。

2026-11-12 增加了 [Kirin9030 单图离线视觉](qwen-offline-vision.md)，完整 ViT 与三路 DeepStack merger 使用原始视觉参数的 FP16 版本，图与权重分离；原在线 W8 视觉和 CPU LLM/EAGLE 文件保留。新增模式需配套更新 MNN 库、插件头及 App 离线桥接源码后重新编译；尚未真机验收。

## 1. 模型来源与精度配置

| 部分 | 来源与处理 |
| --- | --- |
| 语言主干 | [MNN 官方 MNN 包](https://modelscope.cn/models/MNN/Qwen3-VL-2B-Instruct-Eagle3-MNN)，直接保留已有 W4/group64 文件 |
| 共享 embedding / LM head | 同一官方包中的 W8/group64 权重，保留 `tie_embeddings` 原始文件偏移 |
| EAGLE transformer 与特征投影 | 同一官方包中的 W4/group64 权重，直接保留 |
| 视觉编码器 | 从 [Qwen3-VL-2B-Instruct 原始模型](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct) 重新导出；主要线性层用 HQQ W8/group128 压缩 |
| KV Cache | `attention_mode: 8`，保持浮点 KV；CPU `precision: low` 在支持的 ARM 内核上选择 FP16，主机验证使用 FP32 |

官方 MNN 包已经包含可执行的量化主模型和 draft，因此本次不需要重新生成 GPTQ 权重。官方文件没有提供足以确认其完整量化校准过程的信息，这里不把它标记为重新做过 GPTQ。HQQ 是视觉导出阶段的权重量化，没有执行依赖图片数据集的激活校准。参考图片只用于量化后的精度验证，没有参与 HQQ 量化参数求解。

W8 描述视觉 MNN 文件中的权重存储。HiAI 在线建图会读取、转换权重，最终执行精度由后端和设备编译器决定；不能据此称为 NPU 原生 W8A8 计算。MNN 图文件与 `.mnn.weight` 配套分发，也不等同于离线 OMC 的外置权重初始化协议。

## 2. 在线 NPU 模型的结构

```text
图片 → CPU 预处理 / visual_pre
     → HiAI chunk 0 → chunk 1 → … → chunk 5
     → CPU visual_post → image embedding + 三组 DeepStack 特征
     → CPU Qwen3-VL 主模型 ⇄ CPU EAGLE-3 draft
     → 文本
```

24 个视觉 blocks 分为 6 段，每段 4 层，以控制在线 IR 建图峰值内存。DeepStack 在第 5、11、17 层取特征，分别由 chunk 1、2、4 输出，按顺序送入 post。并非每一段都输出 DeepStack。

首次运行时在手机上生成设备相关 OM 缓存。分发包只需要 MNN 图和权重，无需在主机上执行 DOPT 或 OMG。重新加载缓存仍需设备初始化；是否加速以及具体耗时需要真机测量。

导出示例形状为高 608、宽 256。默认配置使用图片实际宽高并对齐到 32；改变尺寸可能触发新的 NPU 编译和缓存。需要固定形状复测时可使用 MNN 图片标签的 `<hw>` 参数。

## 3. 准备源模型并导出

在已经安装本仓库导出依赖的 Python 环境中执行。先用本仓库构建主机转换器：

```bash
cmake -S . -B build/qwen-eagle-host \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON \
  -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
cmake --build build/qwen-eagle-host --target MNNConvert llm_demo --parallel 8
```

下载官方包和匹配的原始模型。本机构建时，下载与生成的模型统一放在 `/temp/models/` 下，避免占用源码目录；当前已准备好的目录见 [models.md](../../models.md)。

```bash
modelscope download --model MNN/Qwen3-VL-2B-Instruct-Eagle3-MNN \
  --revision 297b639db1226f884a15a3f503827e220074f1a3 --local_dir /temp/models/Qwen3-VL-2B-Instruct-Eagle3-MNN
modelscope download --model Qwen/Qwen3-VL-2B-Instruct \
  --revision eb46e64af9b20837a9d6030aedc90f2086e587e9 --local_dir /temp/models/Qwen3-VL-2B-Instruct

python project/harmony/prepare_qwen3_vl_eagle.py \
  --mnn-dir /temp/models/Qwen3-VL-2B-Instruct-Eagle3-MNN \
  --original-dir /temp/models/Qwen3-VL-2B-Instruct \
  --mnnconvert build/qwen-eagle-host/MNNConvert \
  --visual-dir /temp/models/mobiinfer-work/qwen-eagle-new-visual \
  --output /temp/models/Qwen3-VL-2B-Instruct-Eagle3-HiAI-MNN-new
```

输出目录须为空。已经完成视觉导出时，可在同一命令中添加 `--reuse-visual`，使用已有视觉目录组装新的运行包。`manifest.json` 记录文件哈希及语言、draft 文件与官方包一致的检查结果。

## 4. 配置与运行

模型包包含四个入口，共享同一套权重：

| 配置 | 视觉 blocks | 语言生成 |
| --- | --- | --- |
| `config.json` | HiAI 在线 NPU | CPU 主模型 + EAGLE-3 |
| `config_ar.json` | HiAI 在线 NPU | CPU 自回归，用于比较 |
| `config_cpu.json` | CPU | CPU 主模型 + EAGLE-3，用于主机验证 |
| `config_cpu_ar.json` | CPU | CPU 自回归，用于主机验证 |

默认采用 greedy 采样、draft 深度 3、top-k 1、每次最多生成 256 个 token。当前 EAGLE 验证使用 argmax；不要把它改成 penalty/temperature 采样后仍假定保持相同采样分布。

主机示例：

```bash
printf '%s\n' '<img>resource/images/cat.jpg</img>请简短描述图片中的动物。' > prompt.txt
build/qwen-eagle-host/llm_demo /temp/models/Qwen3-VL-2B-Instruct-Eagle3-HiAI-MNN-new/config_cpu.json prompt.txt 64
```

手机端使用 `config.json`，不使用 CPU 验证入口。应用应将模型包整体放在可写目录，使 `npu_cache` 可以持久化；主模型、draft、tokenizer 和视觉文件不能遗漏。首次运行需检查 `[NPU_CACHE] visual_runtime` 的实际后端及设备执行日志，CPU 回退不能算 NPU 验证成功。

应用调用 `response(prompt, stream, nullptr, max_tokens)`，一次传入完整生成预算。逐 token 的 `generate(1)` 和缺少 draft 前缀特征的 KV 续接会使用目标模型自回归，保持执行正确性；这两种调用方式不能用于测量 EAGLE 加速。默认 `reuse_kv: false`，多轮历史可作为完整消息输入重新 prefill。

## 5. 编译鸿蒙库与 App 集成

先按 [SDK 配置](../../project/harmony/sdk_config.sh) 和仓库 README 配置真实 Harmony/OpenHarmony Native SDK 与 HiAI ARM64 库。`HARMONY_HOME` 指包含 `native/build/cmake/ohos.toolchain.cmake` 的 SDK `openharmony` 目录：

```bash
mkdir -p build/qwen-eagle-ohos
cd build/qwen-eagle-ohos
HARMONY_HOME=/path/to/sdk/openharmony \
  bash ../../project/harmony/build_64.sh \
  -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON
```

本次产物为 ARM64、OpenHarmony API 20，开启 LLM、Omni、低内存量化、Transformer 融合、HiAI 与 chunk OM 缓存。EAGLE 已在 LLM 库中编译，无需额外的 EAGLE 动态库，也无需已移除的 `MNN_CPU_WEIGHT_DEQUANT_GEMM` 开关。

提供的 native 包按 App 的 `entry/` 目录组织：

- `entry/libs/arm64-v8a/`：新的 `libMNN.so` 和配套 HiAI 库。
- `entry/src/main/cpp/include/`：对应版本的 MNN、LLM 公共头文件。
- `entry/src/main/cpp/offline_npu2/`：与库匹配的 `llmconfig.hpp`、`ujson.hpp`，供既有混合桥接代码使用。
- `bin/llm_demo`：手机端命令行验证程序。

将这些文件同步到 [App 仓库](https://github.com/doulujiyao12/mobiinfra-oh.git) 后，重新打包并安装 HAP。单独启用 EAGLE 的改动位于 MNN 库，没有新增 NAPI 接口；**新增单图离线模式还修改了编入 `libentry.so` 的桥接源码，因此必须同步 `offline_npu2` 目录、匹配头文件和新 `libMNN.so`，重新编译整个 App**。否则旧桥接只接受 `vision_v1`，读取新 `vision_v2` 会在 NNRT 加载前失败。配套包不含预编译 `libentry.so`，没有安装或更改手机上的 App。已有 Mobi 模型与本次 Qwen 模型须使用不同模型目录和缓存目录。

## 6. 本次验证结果与边界

已完成真实源权重下载及哈希校验、主机工具编译、鸿蒙 ARM64 交叉编译、主机图文推理，以及官方语言/draft 文件的逐文件哈希对照。

固定猫图、高 608 × 宽 256 的视觉数值验证：

| 输出 | FP32 原始视觉模型 vs 实际 W8 MNN：余弦相似度 | 相对 L2 误差 |
| --- | ---: | ---: |
| 最终图像特征 | 0.998802 | 4.90% |
| DeepStack 0 | 0.999955 | 0.96% |
| DeepStack 1 | 0.998702 | 5.10% |
| DeepStack 2 | 0.999392 | 3.49% |

转换前的分块 PyTorch 映射同时核对最终特征与三组 DeepStack，相对 L2 误差均低于 `1.1e-5`。主机图像问答中，EAGLE 与自回归均正确描述出橙白色小猫、石头和绿色草地。

这些是有限样例的正确性与数值检查，不是完整视觉评测集的质量结论。量化主模型在单 token 与批量树验证时可能产生不同 logits，本次长文本生成也出现不同的贪心续写；尚未证明 EAGLE 与自回归逐 token 完全一致。不能把未改动官方权重等同于执行误差为零。

未连接手机，未验证本次模型的 HiAI 在线编译成功率、OM 缓存命中、设备内存峰值和端侧加速比。主机测试耗时不能替代手机性能数据，EAGLE 也不保证在所有 CPU 和提示词上更快。

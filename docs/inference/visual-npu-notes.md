# MobiMind 视觉 NPU（Kirin9030）构建与排查记录

> 最新状态（2026-10-08）：ViT W4A16/group64 发布包获用户真机反馈，能够区分不同图片，暂按图文功能可用记录，详见第 13 节。原 W8A8 及新增输出 INT16 的 W8A8 候选仍未修复，相关调优暂停；其核验以 [W8A8 排查记录](w8a8-probe.md) 为准。早期按参数文件体积推断 per-group 的结论已撤回，见第 12 节。

> 本文是**记录性文档**，归档历次构建的实测数据、失败排查与结论。
> 可执行的流程指南见 `CLAUDE.md` 的「OMG/OMC 离线 NPU 图编译」章节。
>
> 阅读约定：**「已验证」= 已收集真机数据；「用户真机初测」= 用户反馈，未收集详细日志或数值；「实测」= 本机（x86）可复现；「推断」= 仅有间接证据。**

---

## 1. 当前有效配置（结论先行）

| 项 | 值 |
|---|---|
| 目标芯片 | Kirin9030 (V311) |
| 离线 OM 精度 | **ViT W4A16/group64 + `compress_conf`：用户真机初测可用**；FP16 无 `compress_conf` 保留为浮点基线 |
| OM 体积 | W4A16 约 31.99 MB / chunk；FP16 约 101.8 MB / chunk |
| 视觉 chunk 数 | 6（全 NPU），24 blocks 均分，每 chunk 4 层 |
| 输入 shape | `hidden_states_in:1,608,1024` / `rotary_pos_emb:2,608,1,64` / `attention_mask:1,608,608` |
| `.mnn` 侧精度 | visual blocks 为 int8 GPTQ；LLM 为 GPTQ W8A8（CPU） |

**不要**在 Kirin9030 上用 `compress_conf`（W8A8）生成视觉 chunk 的离线 OM —— 见 §2。
该限制针对已失败的 W8A8 配置；已发布 W4A16 使用不同的位宽、分组和 UINT4/LUT 权重编码，见 §13。当前反馈不代表数值严格对齐或 W8A8 问题已修复。

---

## 2. 本模型的 W8A8 `compress_conf` 在 Kirin9030 上精度异常

### 2.1 真机症状

W8A8 `compress_conf` 编出的 6 个 `.om` 在真机上**编译与执行都成功**（`RunSync` 全部返回成功、输入元素数匹配、无 NaN/Inf、无 CPU fallback），但**激活数值完全错误**：

| chunk | 校准数据 absmax（应然） | 真机实测 | 比值 |
|---|---|---|---|
| 0（输入） | 32.28 ~ 33.13 | 32.65 | ✅ 吻合 |
| 1 | **18.89 ~ 24.02** | **152.75** | **7.5×** |
| 2 | 10.88 ~ 14.41 | 151.625 | ~12× |
| 3 | 327.5 ~ 412.5 | 241.0 | 偏低 |
| 4 | 360.75 ~ 444.5 | 272.75 | 偏低 |
| 5 | 379.25 ~ 464.5 | 282.75 | 偏低 |

表现：**回答与图片内容无关，换不同图片回答仍相似**（例如问"照片里是什么"，答"一堆各色的西兰花"）。

### 2.2 关键判据：量化误差有界，不可能产生 7.5× 放大

权重量化本身是健康的（实测）：

```
chunk0 q_proj: fake_quant vs 原始 fp32
  maxdiff = 2.045e-03   量化步长 = 4.122e-03   半步长 = 2.061e-03
  => 误差落在半步长内 ✅
```

**结论：只能是「量化参数被错误套用」，而非精度损失。**

### 2.3 host 侧模拟（torch 注入 DOPT 记录的真实参数）

用 2 张内容不同的图，逐 chunk 前向，观察 `absmax` 与**跨图余弦**（越低越能区分图片）：

| 模拟模式 | chunk0 输出 absmax | 跨图 cos | 判定 |
|---|---|---|---|
| `correct`（正确套用 DOPT 参数） | 20.13 | **0.756** | 能区分图片，符合预期 |
| 退化为固定范围 `[0,1]` | 57.31 | **0.9987** | **图文无关**，与真机同症状 |
| 退化为固定范围 `[-1,1]` | 22.45 | **0.9996** | **图文无关**，与真机同症状 |
| 漏减 offset | 1304.3 | — | 量级远超真机 |
| 当有符号 int8 | 4919.9 | — | 量级远超真机 |
| 直接输出整数 | 44235.7 | — | 量级远超真机 |

真机症状属于「参数被错用 / 退化为固定范围」这一类。

### 2.4 ONNX 无罪（排除导出环节）

用 int8 `.mnn` 引擎 dump 出的固定输入（`hidden_states_in` absmax=32.6131），比较 chunk0 输出：

| 实现 | chunk0 输出 absmax | vs torch |
|---|---|---|
| torch fp32（参考） | 17.2826 | — |
| pathA ONNX（编译 `.om` 用的图） | 17.2832 | cos **0.99977** |
| fp16route ONNX（修复用） | 17.2826 | cos **1.00000** |

4 张图 × 6 chunk 全部 cos=1.0000（残差 ≤0.025，为 fp16 舍入）。**偏差出在 OMG 编译/OM 执行环节。**

### 2.5 可疑日志：96 条 `can't find key [shape]`

| 配置 | 警告数 / chunk |
|---|---|
| 无 `compress_conf`（FP16） | **0** |
| `compress_conf`（act_bit=8） | **96** |
| `compress_conf`（act_bit=16） | 96（且编译失败） |

`96 = 24 层 × 4 字段`（chunk0/1/5 一致），对应 24 个被激活量化的 Linear。即 OMG 的
`quantize_cfg_parser.cpp GenerateIntArrayConfig(95)` 未能完整吃掉 DOPT 的 `quant_params_file`。

**尝试消除该警告，全部无效**（96 条纹丝不动）：

| 尝试 | 结果 |
|---|---|
| `quant_param_2=True` | 仍 96 |
| ONNX 补 `shape_inference`（value_info 0 → 299） | 仍 96 |
| `--fp16` 把 MatMul 权重转 FLOAT16 以匹配 `--weight_data_type` | 仍 96，且编译失败 |
| 设 `custom_group_size` 等环境变量（官方 `opt_main.py` 的做法） | 粒度不变 |

→ 指向 **OMG 侧解析不匹配**，不是我们漏了元数据。

### 2.6 配置矩阵实测（`PLATFORM=kirin9030`）

| 配置 | 编译 | OM 体积 | 96 条警告 |
|---|---|---|---|
| FP16 无 `compress_conf` | ✅ 成功 | 101.8 MB | 0 |
| `compress_conf` act_bit=8 | ✅ 成功 | 52.1 MB | 96 |
| `compress_conf` act_bit=16 | ❌ 失败 | — | 96 |
| `compress_conf` act_bit=16 + unsigned | ❌ 失败 | — | 96 |

`act_bit=16` 失败的确切错误：

```
E/AI_NPUCL: quantize_optimizer.cc QuantizeOptimizer(28)::"QuantizeOptimizer Fail!"
E/AI_FMK : anchor_utils.cpp GetFormat(25)::"The input data anchor is invalid."
E/AI_NPUCL: ascendc_store.cc CloneNodeEdge(23)::"node has no peerOutAnchor."
```

### 2.7 W8A8 从来不是 Kirin9030 的验证配置

仓库内**全部 4 个** `run_*.sh` 都硬编码 `PLATFORM=kirin9020`（`=` 而非 `:-`，会覆盖环境变量）：

| 脚本 | PLATFORM |
|---|---|
| `run_real_calib_256.sh` | kirin9020 |
| `run_real_calib_256_W8A8.sh` | kirin9020 |
| `run_all_chunks_real_calib.sh` | kirin9020 |
| `run_all_chunks_real_calib_W8A8.sh` | kirin9020 |

仓库内部评审早已标注该矛盾，但从未解决：

> `docs/change-summary-20260815.md` §9.3：
> 「`run_all_chunks_real_calib_W8A8.sh` 的 `PLATFORM=kirin9030`：该脚本走 DOPT W8A8 +
> compress_conf 路径，与 Kirin9030 的 FP16 OMC 策略**矛盾**……否则需要确认。」

**本机只装了 `kirin9030` 平台插件**，`kirin9020` / `kirinx90` 目录都不存在。按脚本原样（9020）跑会：
脚本打 `platform plugin not found, skip --platform` → 不传 `--platform` → `OMG Generate execute failed`，**无产物**（实测）。

### 2.8 结论

本模型在 Kirin9030 上的 W8A8 `compress_conf` 离线路径属于**未验证组合**：OMG 对 DOPT 量化参数的消费与预期不符，
且缺少可用的真机迭代条件。**不要在 9030 上用 W8A8 `compress_conf` 生成视觉 chunk 的离线 OM。**

若将来仍要启用 W8A8，需要补齐以下之一：

1. **真机逐 chunk 数值比对** —— 用 app 侧 `runOmVsMnnRealCalibTest` 拿 OM vs MNN-CPU 的 hidden 输出。
   重点看各 chunk 输入 absmax 是否落在上表 §2.1 的校准范围。这能把范围从 OMG 黑盒缩到具体算子。
2. **一份华为官方跑通的 `compress_conf` 样例** —— 与我们的做结构对照，才能判定是「我们生成错了」
   还是「OMG 版本问题」。

### 2.9 附：`act_bit` / `group_size` 是否生效

`act_bit` **在 DOPT 层是生效的**（`quant_params_file` 随其变化）：

| 配置 | quant_params_file |
|---|---|
| act_bit=16 | md5 `ad088919718dd614`, 3133471 B |
| act_bit=8 | md5 `ea01f4cd948c0b26`, 3136670 B |

但 `group_size` **实测不生效**：`--group_size 128` 与 `--group_size 0` 产出的
`quant_params_file` **字节完全相同**（md5 一致）；`weight_quantizer.s` 元素数始终等于
`out_channels`（per-channel），而非 `out_channels × (in_channels/128)`。

> 注：per-channel 比 per-group **更细**，这条本身不会导致图文无关，**不构成根因**。
> 但它说明「`dopt_config` 里写了 ≠ 生效」，不宜再据此推断。

附带一个易误读点：`export-fp16` 产出的 `export_report.json` 里显示 `act_bit: 16, weight_bit: 4`，
这只是 argparse 的**默认值**（命令行没传），从未参与计算。不要当成"FP16 用了 A16/W4"。

---

## 3. DeepStack 输出合并导致真机闪退

### 3.1 现象

6 个 OM 全部执行成功，但紧接着 App 闪退，ArkTS `try/catch` 接不住。

### 3.2 根因

某个 chunk 的 deepstack 层若**正好是该 chunk 的最后一层**，其 deepstack 输出与 `hidden_states`
是**同一个张量**。ONNX 导出时用 `Identity` 给它命名，而 OMG 会把这个冗余 `Identity` 删掉：

```
W/AI_FMK onnx_pre_checker.cpp PreCheckGraph(143)::"the node Identity dont have output tensor"
```

于是该 chunk 的 `.om` **只返回 1 个输出**（其他 deepstack chunk 返回 2 个）。

本模型 `deepstack_visual_indexes = [5, 11, 17]`，24 blocks / 6 chunks → 每 chunk 4 层：

| chunk | blocks | 本 chunk 的 deepstack | 是否 = 末层 |
|---|---|---|---|
| 0 | [0..3] | — | — |
| 1 | [4..7] | [5] | 否 |
| **2** | **[8..11]** | **[11]** | **★ 是（被合并）** |
| 3 | [12..15] | — | — |
| 4 | [16..19] | [17] | 否 |
| 5 | [20..23] | — | — |

**只有 chunk2 需要补。**

### 3.3 为什么直接闪退

`allDeepstack` 少一路 → `visual_post` 收到 3 个输入而非期望的 4 个 →
`MNN_ASSERT` 在 release 下是**空宏**（`include/MNN/MNNDefine.h:52`），拦不住 → 越界。

### 3.4 修复

`config.json` 补：

```json
"visual_blocks_om_deepstack_dup": [2]
```

该字段取值**只取决于 chunk 划分与 deepstack 索引，与校准来源无关**，两套校准都需要。

**安全性质**：引擎侧触发条件是 `mOmDeepstackDupIndices.count(i) && omOutputs.size() <= 1`，
所以即使误配也不会重复补（OM 返回 2 个输出时自动跳过）。

### 3.5 判定方法

遍历各 chunk ONNX 的 `graph.output`：若 deepstack 输出名由 `Identity` 产生、且其输入
== `hidden_states` 的输出名，则该 chunk 需要 dup。

### 3.6 为什么 host 测不出来

host 上无 hiai NPU，验证时 `visual_blocks_chunk_backends` 全置 `cpu`，走 **MNN module** 路径
—— 它保留 chunk2 的 2 个输出。**只有真机 OM 路径才会触发。**

---

## 4. 校准脚本漏掉 `pos_embeds`

### 4.1 根因

`visual_pre.mnn` 的 `hidden_states` 输出是 **`patch_embed + pos_embeds`**
（见 `llmexport.py::_build_visual_split_wrappers._VisualPre`，及 HF
`Qwen3Vision.forward` 的 `pos_embeds = pos_embed(idx)*w; hidden += sum(pos_embeds)`）。
它同时是 chunk0 的输入。

`generate_npz_calib.py` 早期只写 `patch_embed`，漏了 `pos_embeds`。

### 4.2 实测（同图，hw=600,270）

| 计算 | absmax |
|---|---|
| 仅 `patch_embed`（bug 版） | 5.747 |
| `patch_embed + pos_embeds`（真值） | **32.618** |

漏项使 chunk0 输入被低估约 5.7 倍。**该 bug 不报错**，OMC 照样编译成功，只能靠数值比对发现。

### 4.3 修复与验证

`patch_embed` 前先 `view(N, -1)`，再按 `get_idx_weight(grid_thw)` 加 `pos_embeds`。

验证：chunk0 `hidden_states_in` absmax 应 ≈ **32.6**（旧 bug 版 ≈5.7）；
chunk1 absmax 均值应 ≈ **17.3**，与设备端日志 `hidden absmax=17.4068` 吻合（独立交叉验证）。

---

## 5. 路径 A vs 路径 B（校准数据来源）

| | 路径 A：引擎 dump | 路径 B：纯 Python 前向 |
|---|---|---|
| 依赖 | `llm_demo`（需开 `MNN_VISUAL_CHUNK_INPUT_DUMP`）+ **6chunk MNN 模型目录** | 仅 HF fp 模型 |
| 脚本 | `select_images.py` → `llm_demo` → `bin_to_chunk_npz.py` | `generate_npz_calib.py` |
| 图片选取 | `rng.sample(seed=42)`（随机） | `sorted(files)[:N]`（非随机） |
| 现状 | **6chunk 模型目录是外部依赖，已不存在** | **推荐** |

两条路径产出的 npz **格式完全一致**，可直接互换；但：

> **"格式一致" ≠ "数值等价"**。
> 历史上正是因为只对比了 shape/dtype 就认定两者等价，才漏掉了 §4 的 `pos_embeds`。
> **切换路径或修改任一路径的前向后，务必做数值交叉验证**（比对 chunk1 absmax 是否 ≈17.3）。

### 5.1 两条路径的数值差异（2026-10 实测）

同一张图：

| chunk | A（引擎 dump） | B（torch fp32） | A/B |
|---|---|---|---|
| 0 | 32.625 | 32.625 | 1.0000 |
| 1 | 22.000 | 18.781 | 1.171 |
| 2 | 13.820 | 11.438 | 1.208 |
| 3 | 382.000 | 335.750 | 1.138 |
| 4 | 412.500 | 369.000 | 1.118 |
| 5 | 432.000 | 387.750 | 1.114 |

chunk0 逐元素一致（maxdiff 0.016，fp16 舍入），差异全部出现在**过完 block 之后**。

三方隔离实验（seq_len=256，起点为引擎 dump 的 chunk0 输入）：

| 实现 | chunk0 输出 absmax | vs torch |
|---|---|---|
| torch | 18.1250 | — |
| llmexport ONNX（onnxruntime 执行） | 18.1250 | +0.00% |
| llmexport `.mnn`（引擎执行） | 19.6875 | **+8.62%** |

用 plugin-route 的 6 个 ONNX 串成级联链（≈真机 NPU 轨迹）与 torch 对齐（差 1~2%），
而路径 A（MNN CPU 轨迹）偏高 11~21%。

**已排除的假设**：不是 fp16 精度（torch 全程 fp16 与 fp32 几乎相同）；不是权重量化
（换成 GPTQ 反量化权重后 18.717 vs fp 18.783，几乎无变化）。
**未定位**：MNN block 执行偏离 torch 8.62% 的具体算子。

> 由于部署链（`.om` 来源）是 plugin-route ONNX 谱系，**路径 B 与部署链自洽**。

---

## 6. 引擎侧加固（`omni.cpp` / `PipelineModule.cpp`）

与本项目两次真机问题配套的防御性改动：

| 文件 | 改动 |
|---|---|
| `omni.cpp::qwen2VisionProcess` | 调用 `visual_post` 前校验 `blocksOut.size() == postInfo->inputNames.size()`，不匹配则 `MNN_ERROR` 并返回空 |
| 同上 | `outputs` 为空时提前返回，避免 `outputs[0]` 越界 |
| `PipelineModule::onForward` | `mInputSize != inputs.size()` 时打印明确错误并返回空；给 submodule 输入/输出 stack 索引加范围检查 |

验证：故意构造会少一路 deepstack 的 config —— 旧二进制越界崩溃，新二进制打印
`visual_post input count mismatch: got 3, expected 4` 并**优雅退出**（exit 0）。

> 这些守卫**不是**修复手段，只是让同类配置错误报错而非崩溃。
> 注意 `express/module/PipelineModule.cpp` 静态链入 `libMNN.so`（App 侧无独立
> `libMNNExpress.so`），所以守卫需重编才能生效；仅改 config 时不需重编。

---

## 7. 历史遗留（已过时，仅供追溯）

### 7.1 DDK 6.0.1.0 的 W8A8 编译失败

旧 DDK 6.0.1.0 下，W8A8 `compress_conf` 会令 `MatMulV2` 按 FP16 权重大小校验，而压缩 INT8 buffer
只有预期一半，报 `Size check failed. realSrcSize < expectSrcSize` → `Trans weight failed`。
DDK 6.1.1.0 修掉了这个**编译期**报错 —— 但**数值问题（§2）依然存在**，不要混淆这两件事。

### 7.2 失效路径（重启后已不存在）

`/temp/huawei-sdk/DDK-tools-next-6.0.1.0`、`/temp/models/csm/`、`/temp/fdh/`、`/temp/csm/`
均已在 2026-10 重启后消失。`run_*.sh` 中仍有引用，**直接跑会失败**。

### 7.3 Kirin9020 路径

`run_*.sh` 与 README 示例都面向 kirin9020（DOPT W8A8 + `compress_conf`，无需 AscendC）。
**本机无 9020 插件，该路径从未在此环境验证过。**

### 7.4 发布到 ModelScope 的历史仓库

| 仓库 | 内容 | 状态 |
|---|---|---|
| `fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_fp16om` | FP16 离线 OM，6 chunk | **真机验证可用** |
| `fengerhu1/mnn_mobi_2B_w8a8_visual6_npu_kirin9030_enginecal` | W8A8，引擎 dump 校准 | ⚠️ 数值错误 |
| `fengerhu1/mnn_mobi_2B_w8a8_visual6_npu_kirin9030_torchcal` | W8A8，torch 校准 | ⚠️ 数值错误 |
| `fengerhu1/mnn_mobi_2B_w8a8_visual_npu_kirin9030` | 早期 FP16 版（4 NPU chunk） | 历史 |

---

## 8. host 侧验证的边界（重要）

**host 上能测的**：ONNX 数值（vs torch）、MNN `.mnn` 加载与前向、视觉全链路（全 CPU 后端）、
校准 npz 的统计特征。

**host 上测不到的**：
- **OM 的输入索引顺序** —— README 明确警告 OMG 会重排输入张量
- **OM 的输出个数与顺序**（尤其 chunk 1/2/4 的双输出）
- **OM 的实际数值** —— 无 x86 侧 OM 运行时
- **`compress_conf` 在真机上的行为** —— `eval_chunk_quant.py` 不经过 OMG

> 本项目 README 里「W8/A16 + 真实校准 cos>0.9997」的结论是 host 侧测的，
> **没有覆盖 OMG `compress_conf` 的真机行为** —— 这正是「host 全绿、真机错」的来源。

---

## 9. 实验 2：用 W8A8 的 ONNX，但不加 `--compress_conf`

**目的**：把「OMG 的 `compress_conf` 量化 pass」与「DOPT 产出的 fake-quant 权重」两个嫌疑切开。

### 9.1 三组对照

| | OM 的 ONNX 来源 | OMG `--compress_conf` | 真机 |
|---|---|---|---|
| A. FP16 包 | 原始 HF fp32 → fp16 | 无 | ✅ 正常 |
| **B. EXP2（本实验）** | **DOPT fake-quant（与 C 同一份文件）** | **无** | ⏳ 待验证 |
| C. W8A8 包 | DOPT fake-quant | **有** | ❌ 数值错误 |

判读：

- **B 正常** → 根因在 OMG 的 `compress_conf` 量化 pass（或其消费的 `quant_params_file`）
- **B 错误** → 根因在 fake-quant 权重本身（DOPT 侧），与 OMC 无关

### 9.2 实验设计（关键：单一变量）

以**真机已验证可用的 FP16 包为基座，只替换 `om/`**：

- `config.json`、`llm.mnn`、`llm.mnn.weight`、`visual_*.mnn/.weight`、`tokenizer.mtok`、
  `embeddings_bf16.bin` —— **22 个文件与 FP16 包逐字节相同**（实测 md5 一致）
- 仅 `om/` 与两个元数据文件（`SHA256SUMS`、`offline_om_manifest.json`）不同

因此真机上唯一的变量 = **该 `.om` 由哪个 ONNX 编译而来**。

### 9.3 ONNX 复用情况（已验证）

| 检查 | 结果 |
|---|---|
| EXP2 的 ONNX vs W8A8 的 ONNX | md5 **相同**（`06b9d2880194a9cc`）✅ 确实是同一份 fake-quant 图 |
| EXP2 的 ONNX vs FP16 的 ONNX | md5 不同（`093d9355e93365c1`）✅ 确非同一份 |
| MatMul 权重 dtype | `FLOAT`（24 个）= fake-quant 值 |

### 9.4 编译结果（本机实测）

6 个 chunk 全部通过判据：

```
partition type NPU:1, CPU:0 | save=1 | succ=1 | fallback=0
```

| 指标 | EXP2 | FP16 | W8A8 |
|---|---|---|---|
| `.om` 体积/chunk | **101.76 MB** | 101.77 MB | 52.12 MB |
| `can't find key [shape]` 警告 | **0** | 0 | 96 |
| `Fuse Cube Type`（QBM 融合） | **0** | 0 | 24 |
| `.om` 内量化算子 | **无** | 无 | `QuantBatchMatmulV3` / `StaticQuant` / `QuantizeV2` |
| deepstack 结构 | 同 FP16 | — | 同 |

OMG 实际命令行：`--weight_data_type=FP16 --target=omc --platform=kirin9030`（**无 `--compress_conf`**），
与 FP16 路线完全一致。三版 `.om` 的 md5 互不相同（确认权重确实参与了编译）。

> 即：EXP2 走的是**非压缩 FP16 图**编码路径，但权重内容是 int8 量化后的值。

### 9.5 一个支持性观察（host 侧）

fake-quant 权重经 fp16 存储仍能保住 int8 网格：

| 层 | int8 量化步长 | fp16 舍入误差 | 比值 |
|---|---|---|---|
| `q_proj` | 0.004122 | 0.000122 | 0.030 |
| `linear_fc1` | 0.002568 | 0.000122 | 0.048 |

比值 ≪ 1，说明 **fp16 精度足以无损承载 int8 网格**。因此 EXP2 的 `.om` 里权重
≈ W8A8 的权重；两者若结果不同，差异只能来自 OMG 是否施加 `compress_conf` 的激活量化。

### 9.6 产物位置

```
/temp/work/exp2_rel/          # 交付包 (31 文件 / 3.7 GB)，可直接推真机
  om/visual_blocks_npu_0..5.om   # 101.8 MB/chunk
/temp/work/exp2route/chunk<i>/   # 编译中间产物
```

已上传 ModelScope（公开）：

- **EXP2**：`fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_exp2_nocompressconf`

对比用另外两包：`/temp/work/fix_rel`（FP16，已知正常）、
`/temp/models/mnn_mobi_2B_w8a8_visual_npu_kirin9030_pathA`（W8A8，已知错误）。

### 9.7 真机验证步骤

1. 用 `/temp/work/exp2_rel/` 整目录替换手机上的模型包
2. 确认 6 个 chunk 的 `RunSync` 成功
3. **打印各 chunk 的 hidden 输入 min/max**，对照 §2.1 的校准范围
4. 用多张不同图片提问，确认回答随图片变化

### 9.8 结论（真机已验证，2026-10）

✅ **EXP2 在真机上验证正确**（回答与图片内容相符、换图后回答随之变化）。

由此**根因锁定**：

| 包 | ONNX 来源 | `compress_conf` | 真机 |
|---|---|---|---|
| FP16 | 原始 HF 浮点 | 无 | ✅ 正常 |
| **EXP2** | **DOPT fake-quant** | **无** | ✅ **正常** |
| W8A8 | DOPT fake-quant | 有 | ❌ 数值错误 |

- EXP2 与 W8A8 用的是**同一份 ONNX**（md5 相同）→ **fake-quant 权重无罪**
- EXP2 与 W8A8 的唯一差别是 `--compress_conf` → **根因在 OMG 的 `compress_conf` 路径**

**重要：EXP2 没有体积收益。** 它 101.7 MB/chunk，与 FP16 相同 —— 因为不加 `compress_conf` 时
OMG 按 `--weight_data_type=FP16` 存放权重（2 字节），fake-quant 的 int8 值仍占 2 字节。

因此下一步的修复目标不是"用 EXP2 替代"，而是：**拿到 int8 体积（≈52MB）又不触发坏掉的激活量化**。
候选方向见 §10。



---

## 10. 尝试修复 `compress_conf`（EXP3/4/5，均未成功）

### 10.1 目标

§9 已证明根因在 OMG 的 `--compress_conf` 路径。修复目标是：
**拿到 int8 体积（≈52MB/chunk）又不触发坏掉的激活量化**。

体积收益的量化（供权衡）：

| | om/ 合计 | 整包 | 备注 |
|---|---|---|---|
| FP16 / EXP2 | 611 MB | 3.70 GB | |
| W8A8 | 313 MB | 3.40 GB | **仅省 8%** |

非 `om/` 部分（3.1 GB：`llm.mnn.weight` 2.1 GB + `embeddings_bf16.bin` 0.6 GB + 其他）**两种方案完全相同**。

### 10.2 尝试清单（全部未改变 OMG 的量化行为）

| # | 修法 | shape 警告 | QBM | QuantizeV2 | 结果 |
|---|---|---|---|---|---|
| — | **基线 W8A8** | 96 | 24 | 382 | 数值错误 |
| — | **对照 FP16（无 compress_conf）** | **0** | **0** | **0** | 正常 |
| 1 | `quant_param_2=True` | 96 | 24 | 382 | 无效 |
| 2 | `act_bit=16` | 96 | 5 | — | 编译失败 |
| 3 | `act_bit=16` + `--input_unsigned_quant` | 96 | 5 | — | 编译失败 |
| 4 | 删 `dopt_config` 的 `input` 段 | 96 | 24 | 382 | 无效 |
| 5 | ONNX 补 `shape_inference`（value_info 0→299） | 96 | 24 | 382 | 无效 |
| 6 | ONNX 权重转 FLOAT16 | 96 | 0 | — | 编译失败 |
| 7 | `set_quant_state(input_state=False)` | 96 | 24 | 382 | 无效 |

「无效」= OMG 侧指标与基线逐项相同，即激活量化照常施加。

### 10.3 EXP3：删 `dopt_config` 的 `input` 段

删掉 24 层的 `input` 段（只留 `weight`）后重新校准 + 编译：

```
OM size = 52099056   shape警告 = 96   QBM = 24   QuantizeV2 = 382   StaticQuant = 649
```

与基线一致 → **激活量化不由 `dopt_config` 的 `input` 段决定**。

### 10.4 EXP5：`set_quant_state(input_state=False)`

对齐官方 `calibration.prototxt` 的 `inference_with_data_quantized: false`，
把校准冻结后那次 `set_quant_state` 的 `input_state` 改为 `False`：

```
quant_params_file = 3138327 B  md5=1465c11435682a65
OM size = 52099056   shape警告 = 96   QBM = 24   QuantizeV2 = 382
```

与基线一致 → **`input_state` 也不影响该文件**。

### 10.5 `quant_params_file` 变体汇总

| 变体 | 大小 | md5 |
|---|---|---|
| W8A8 基线（256 样本, act8） | 3139781 | `060f44b138f5d471` |
| EXP3（删 input 段） | 3138327 | `1465c11435682a65` |
| EXP5（input_state=False） | 3138327 | `1465c11435682a65` |
| `quant_param_2=True` | 3138327 | `1465c11435682a65` |
| act16 | 3133471 | `ad088919718dd614` |
| 官方 mnist（参考, 仅 14 节点） | 4002 | `b61b43d8f774fccc` |

关键观察：

- EXP3 / EXP5 / `quant_param_2=True` 三者 md5 **完全相同** → 这三个开关都不影响文件生成
- 文件大小主要随**校准样本数**变化（256 样本 = 3139781 B，16 样本 = 3138327 B）
- 实测体积 ≈ **per-group(128)** 量级（3.14 MB），而非 per-channel（0.29 MB）

### 10.6 EXP4：`generate_quant_params` 不可重复调用

原本想对比「冻结后立即 generate」与「再做一次 `set_quant_state` 后 generate」，
但第二次调用抛 `AttributeError: 'QLinear' object has no attribute 'weight'`
—— 该函数会就地改写模型状态，**不能调用两次**。此对比作废。

（`route` 第 985 行在校准冻结后确实多调了一次 `set_quant_state`，看起来冗余；
但 EXP5 已证明改它的 `input_state` 无效，故**不构成根因**。是否清理此行属代码整洁问题，
不改变行为。）

### 10.7 结论

**本机侧已无法进一步推进。** 7 种修法均未改变 OMG 对激活量化的施加；
`quant_params_file` 的内容也不随这些开关变化。

剩下两种可能性，**都需要 OMG/DDK 侧的权威参照才能区分**：

1. **我们生成的 `quant_params_file` 与 OMG 期望的 schema 不符**（那 96 条
   `can't find key [shape]` 是直接信号）。`quant_params_file` 是私有二进制，
   无法直接核对；需要一份「华为官方流程产出的、在 9030 上真机验证通过的」`compress_conf`
   做结构对照。
2. **OMG 6.1.1.0 在 Kirin9030 上的 `compress_conf` 路径本身有缺陷**
   （该组合从未被真机验证过，见 §2.7）。

### 10.8 建议

**当前生产方案仍用 FP16（`..._fp16om` 仓库）**，它是唯一经真机验证可用的配置。

W8A8 仅省 8% 整包体积（298 MB），却带来数值错误风险 —— **性价比不足以继续投入**，
除非有明确的体积硬约束。

若仍要继续排查，需要的输入（本机无法自行获得）：

- 一份官方 DOPT → OMG 流程产出的、真机验证可用的 `compress_conf`（用于结构对照）
- 或 Huawei 对 `quantize_cfg_parser` 的 schema 说明
- 或真机上逐 chunk 比对 OM vs MNN-CPU 的 hidden 输出（定位到具体算子）

---

## 11. 校准正确性验证 + 官方资料检索（EXP6/7，2026-10）

### 11.1 问题：EXP2 会不会是「OMG 忽略了 fake-quant，用了原始权重」？

**实测排除**（读 ONNX 的 initializer，ONNX 会去读同目录 `.pb` 外部数据）：

| 对比 | maxdiff | 判定 |
|---|---|---|
| ONNX 权重 vs DOPT `fake_quant_weight.pth` | **0.000e+00** | ★ 完全一致 |
| ONNX 权重 vs 原始 HF fp32 | 2.045e-03 | 不同 |

且 OMG 命令行只接收 `--model <onnx>`，**没有第二个权重文件来源** ——
它不可能拿到原始 HF 权重。→ **OMG 用的是 fake-quant 权重。**

### 11.2 但 EXP2 有盲区（重要）

EXP2 没给 `compress_conf`，OMG 按 `--weight_data_type=FP16` 存权重（2 字节）：

| | EXP2（无 compress_conf） | W8A8（有 compress_conf） |
|---|---|---|
| 权重存储 | fp16，2 字节 | **int8，1 字节** |
| 权重反量化 | 无 | **有** |
| 激活量化 | 无 | **有**（QuantizeV2/StaticQuant） |
| `.om` 体积 | 101.8 MB | 52.1 MB |

所以 EXP2 只证明了 **fake-quant 的数值没问题**，**没有**验证
「int8 打包/反量化」与「激活量化」这两条只有 `compress_conf` 才走的路。
§9 的推论「根因在 compress_conf 路径」依然成立，但不能进一步断定
「既然 EXP2 正常，权重侧一切都好」。

### 11.3 量化校准本身是健康的（逐层验证）

**权重量化**（fake-quant 是否落在 int8 网格上）：

```
24/24 层 精确落在网格上 (|frac| 最大 7.63e-06)
  blocks.0.mlp.linear_fc1  w=(4096,1024)  step=0.002548  frac=7.63e-06
  blocks.0.self_attn.q_proj w=(1024,1024) step=0.004089  frac=7.63e-06
```

**激活量化**（参数是否自洽，`s == (max-min)/255`）：

```
24/24 层 精确自洽
  blocks.0.mlp.linear_fc1  min=-30.2545 max=33.7164 bit=8  s期望=0.250866 s实际=0.250866 ✅
```

→ **校准没问题**（这与 §2.2 的「误差在半步长内」互相印证）。

### 11.4 官方资料检索结果（历史状态：当时无法获取）

后续已取得华为插件式量化、三段式量化及量化效果评估说明，并结合官方 serializer 核验参数；具体结论见 [后续排查记录](w8a8-probe.md)。本节仅保留当时检索状态。

按要求尝试获取华为官方 `compress_conf` 文档/样例，结论是**拿不到**：

| 途径 | 结果 |
|---|---|
| DDK 内文档（`*.md`/`*.txt`/`*.pdf`） | 全盘搜索无 `compress_conf` 相关说明 |
| DDK 内样例 | 仅 `tools_omg/sample/calibration.prototxt`（**不同格式**的量化配置，非本文件） |
| 含 `compress_conf` 字样的文件 | 仅 2 个 dopt demo 的 `run_release.sh`（只有调用方式，无格式说明） |
| 网络检索 | **两次检索均无有效结果返回** |

`quant_params_file` 是 DOPT 私有二进制（熵 5.4 bit/byte，有结构但无长度前缀字符串，
非 zlib/gzip/lzma，非文本 protobuf），**无法直接解析**。

### 11.5 新发现的强信号：192 条 error 级 anchor 错误

| 日志项 | W8A8 | FP16 | EXP2 | 官方 mnist | 级别 |
|---|---|---|---|---|---|
| `can't find key [shape]` | **96** | 0 | 0 | 0 | I/ info |
| **`anchor is invalid`** | **192** | **0** | 0 | 24 | **E/ error** |
| `peerOutAnchor` | 126 | 206 | — | 10 | — |

`192 = 24 个量化 matmul × 8 个 anchor`。错误出现在 `fixpipe_pass.cpp SetFixpipeFormat`
之后，且附近必有 `QuantBatchMatmulV3`/`fixpipe` 节点（264 处共现）。
**这是目前 wp8a8 独有的、error 级的异常**，FP16 与 EXP2 均为 0。

### 11.6 另一个数值巧合：scale 比值 ≈ 真机放大倍数

| 量 | 值 |
|---|---|
| `input_quantizer.s` 的 max/min | **7.35** |
| 真机 chunk0 输出放大倍数（152.75 / 20.4） | **7.49** |

两者高度接近，**符合「某层的激活 scale 被错配到另一层」的特征** ——
与 §2.3 的模拟结论（「参数被错用」而非「精度不够」）一致。
但这只是相关性，**不构成证据**。

### 11.7 官方 demo 在 kirin9030 上也失败

用官方 `demo/quant8-8/notrain/pytorch_mnist`（CPU 模式 + protobuf workaround）生成
官方 ONNX + 官方 `compress_conf`，再过 OMG：

```
E/ASC: cannot find a valid kernel configuation for node [name:fc1_fixpipe,type:QuantBatchMatmulV3]
```

→ **官方流程在 kirin9030 上也编不出 W8A8**。因此这个 demo **不能**作为
「官方能跑通」的对照。

> 注：`--hiai_version` 只支持 `master/IR/v310/v300`，而 Kirin9030 是 **V311**；
> 试 `--hiai_version v310` 报 `version dir V310 and v310 not exists`（本机无该目录）。

### 11.8 已尝试的 OMG/DOPT 配置全表（11 种，全部未改变 compress_conf 行为）

| # | 配置 | OM 体积 | shape 警告 | anchorErr | 真机 |
|---|---|---|---|---|---|
| — | FP16 原始权重，无 compress_conf | 101768623 | 0 | 0 | ✅ 正常 |
| — | EXP2 fake-quant，无 compress_conf | 101756141 | 0 | — | ✅ 正常 |
| — | **W8A8 fake-quant + compress_conf** | 52115428 | 96 | 192 | ❌ 错误 |
| 1 | `quant_param_2=True` | 52099056 | 96 | — | — |
| 2 | `act_bit=16` | 编译失败 | 96 | — | — |
| 3 | `act_bit=16 + unsigned` | 编译失败 | 96 | — | — |
| 4 | 删 `dopt_config` 的 `input` 段 | 52099056 | 96 | — | — |
| 5 | ONNX 补 `shape_inference` | 52115428 | 96 | — | — |
| 6 | ONNX 权重转 FLOAT16 | 编译失败 | 96 | — | — |
| 7 | `set_quant_state(input_state=False)` | 52099056 | 96 | — | — |
| 8 | `--hiai_version v310` | 报错（无该目录） | — | — | — |
| 9 | `--use_origin_format true` | 编译失败 | 96 | 192 | — |
| 10 | `--weight_merge false` | 编译失败 | 96 | 192 | — |
| 11 | 官方 `build_params`（stage3） | 失败（`AutoModelForCausalLM` 不认 `Qwen3VLConfig`） | — | — | — |

### 11.9 结论与后续

**已确认**：

1. 校准（权重 + 激活）**健康**，ONNX 里装的确实是 fake-quant 权重
2. 根因在 `compress_conf` 路径（§9 三方对照）
3. 192 条 error 级 `anchor is invalid` 是 W8A8 独有信号
4. 官方文档/样例**拿不到**；官方 demo 在 9030 上**也失败**

**仍无法在本机定位**：`quant_params_file` 与 OMG 期望 schema 的具体差异。

**需要的外部输入**（任一即可继续）：

1. 一份**在 Kirin9030 上真机验证通过的** `compress_conf`（结构对照）
2. 官方对 `quantize_cfg_parser` 的 schema 说明
3. 真机逐 chunk 比对 OM vs MNN-CPU 的 hidden 输出（定位到具体算子）

---

## 12. `quant_params_file` 专项分析（2026-10）

### 12.1 它在链路中的位置（唯一通道）

ONNX **不含**任何量化信息（实测：量化节点 0 个、`metadata_props` 空、
无 `scale`/`zero`/`quant` 命名的 initializer、300 个节点全在 `ai.onnx` 域）。
ONNX 只是"权重值已 fake-quant 的纯浮点图"。

**OMG 拿到激活量化参数的唯一途径就是 `--compress_conf <quant_params_file>`。**

| 产物 | 装什么 | 谁消费 |
|---|---|---|
| `onnx/*.onnx` + `.pb` | 权重**值**（无量化参数） | OMG 读权重 |
| `quant_output/calibrated_chunk_XX.pth` | 权重值 + 每层量化参数（可读） | 调试 |
| `quant_output/fake_quant_weight.pth` | fake-quant 后的权重值 | 导出 ONNX |
| **`quant_output/quant_params_file`** | **权重+激活量化参数（私有二进制）** | **OMG** |

### 12.2 已确证：权重量化是 per-channel

用"`w/s` 是否落在整数网格上"判定（|frac|→0 者为真）：

| 层 | w.shape | per-channel \|frac\| | per-group(128) \|frac\| |
|---|---|---|---|
| blocks.0.mlp.linear_fc1 | (4096,1024) | **7.63e-06** ✅ | 0.50 ❌ |
| blocks.0.mlp.linear_fc2 | (1024,4096) | **7.63e-06** ✅ | 0.50 ❌ |
| blocks.0.self_attn.q_proj | (1024,1024) | **7.63e-06** ✅ | 0.50 ❌ |

→ **fake-quant 权重确实是 per-channel 量化的**，state_dict 里的 `weight_quantizer.s`
（36,864 个 = 各层 out_channels 之和）与实际量化粒度一致。

### 12.3 历史文件体积推测（已撤回）

下方表格是早期按定宽原始数值估算文件大小的推测。后续通过官方 serializer 捕获实际参数并与 checkpoint、fake-quant 权重配对核验，参数符合 per-channel；私有编码文件的体积不能用来判定量化轴或粒度。因此，下方所谓“不一致”不再作为缺陷证据。

| 粒度 | 元素数 | ×8B（scale+offset） | 与实际文件的吻合度 |
|---|---|---|---|
| **per-channel** | 36,864 | 294,912 B | **9.4%** ❌ |
| **per-group(128)** | 393,216 | 3,145,728 B | **99.81%** ✅ |
| 实际文件 | — | **3,139,781 B** | — |

当时将下面的体积对比解释为量化粒度矛盾；该解释已撤回：
- 权重是按 **per-channel** 量化的（§12.2 已确证）
- 但传给 OMG 的参数文件，尺寸是 **per-group(128)** 的量级（差 10.6 倍）

### 12.4 `group_size` 配置实测不生效

| 配置 | quant_params_file md5 |
|---|---|
| `--group_size 128` | `ea01f4cd948c0b26` |
| `--group_size 0` | **`ea01f4cd948c0b26`（完全相同）** |

此外设 `custom_group_size` 环境变量（官方 `opt_main.py` 的做法）也不改变
`weight_quantizer.s` 的粒度。→ **`dopt_config` 里的 `group_size` 没有传到 DOPT。**

### 12.5 文件是变长编码（尺寸随数值变化）

6 个 chunk 的量化层形状**完全相同**，但文件大小不同：

| chunk | 大小 |
|---|---|
| 0 | 3,139,781 |
| 1 | 3,142,904 |
| 2 | 3,145,328 |
| 3 | 3,140,393 |
| 4 | 3,122,713 |
| 5 | 3,126,538 |

跨度 22,615 B（0.72%）→ 说明**大部分是定宽数组 + 少量变长元数据**。

### 12.6 字节混淆特征（未完全解码）

| 检测 | 结果 |
|---|---|
| 原始可打印率 | 59.46%（偏高） |
| 单字节 XOR 后最高可打印率 | **99.98%**（随机基线 37%） |
| 逐字节自相关（raw[i]==raw[i+shift]） | 无显著峰 |
| 每列熵（周期检测） | L=7 时 3.683，显著低于 L=1 的 5.406 |
| IOC(1) / IOC(7) | 7.06 / **24.66** |
| 最佳 XOR 解出可读层名（`blocks`/`linear_fc`/`self_attn`） | **0 命中** |

**结论：文件经过某种字节变换（大概率 XOR 类），但我没能解出可读结构。**
`XOR 0x64` 后 99.98% 可打印，但内容仍是类文本的乱码，搜不到任何层名 ——
不能确认已正确解码，故不作为证据。

### 12.7 无法在文件中定位 scale 值

| 搜索对象 | 命中 | 随机对照 |
|---|---|---|
| state_dict 的 192 个量化参数（激活 + 权重 scale，多种编码） | 12/192（6.2%） | 5/192（2.6%） |

两者仅差 2.4pp，**接近假阳性基线** → 数值不是以明文（fp32/fp16/fp64）
直接存储的，与 §12.6 的混淆特征一致。

### 12.8 后续核验后的结论

原六份压缩参数可以由校准 checkpoint 经官方 serializer 逐字节复现，权重、激活、bias 与输出反量化参数已与 fake-quant/ONNX 配对检查。没有据此发现参数文件损坏或权重量化轴不匹配；原先按文件体积推断 per-group 的结论撤回。

W8 的 `group_size` 属于配置约束问题，省略后原参数仍能逐字节复现，不能将配置修正当作设备精度修复。`compress_conf` 触发的 OMG/Kirin9030 执行路径仍与量化参考不一致，具体根因未定位。新增输出 INT16 的完整候选经用户真机测试仍不对齐，当前已停止继续调优。

以下仅保留当时的后续思路，当前不执行：

1. **拿一份在 Kirin9030 上真机验证通过的 `quant_params_file`** —— 与我们的做
   尺寸/结构对照，能立刻判断是"我们生成错了"还是"OMG 解析问题"
2. 若拿不到，**在能跑通 W8A8 的环境里 dump 中间产物**：把 `generate_quant_params`
   的输入 state_dict 与输出的文件都保留，逐项对照
3. 真机逐 chunk 比对 OM vs MNN-CPU 的 hidden 输出，定位到具体算子

---

## 13. ViT W4A16/group64 真机初测可用（2026-10-08）

### 13.1 当前结论与证据来源

用户下载本次 W4A16 完整模型并在手机运行后反馈：精度问题似乎没有之前严重，至少能够区分不同图片，暂认为能够正常运行。按用户此前允许数值误差、以图文相关性为主的验收标准，将这个**特定发布包**记录为“用户真机初测可用”。

本次反馈未提供图片数量、具体图片与回答、固定输入 probe 输出或 NPU 日志。因此尚未得到 NPU 对同配置 CPU 量化参考的 cosine / relative L2，也未独立确认此次运行的六图加载和 fallback 统计；不能将该反馈写成严格数值对齐、完整准确率通过或根因已定位。原 W8A8 的故障结论保留。

### 13.2 对应产物与量化配置

ModelScope 完整模型包：[`fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_w4a16_g64`](https://www.modelscope.cn/models/fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_w4a16_g64)。发布时 31 个文件的远端大小和 SHA-256 已核对，六张离线图还通过公开下载链接重新下载核验。

| 项目 | 本次配置 |
|---|---|
| 变更范围 | 仅六张 ViT block 离线图；LLM、视觉前后处理及 MNN block 文件保持原样 |
| 工具链 | DDK-tools-next-6.1.1.0 / kirin9030-plugin-next-6.1.1.0 |
| 策略 | `Quant_act_weight_eco` |
| 权重 | W4、group64；`UINT4` 索引及 signed LUT `[-8,7]` |
| 激活 | signed INT16 输入量化、逐通道 INT16 输出量化；A16 不表示始终以 FP16 计算 |
| 校准 | 每段 192 份真实路径 A 输入，group MinMax 初始化 |
| 权重优化 | 本机没有 CUDA，未执行完整 GPTQ/QAT 三段式权重优化 |
| 序列化 | `quant_param_2=True`，重导配套 ONNX 后传入 `compress_conf` |
| 编译 | `--target=omc --platform=kirin9030`，六段均为 `NPU:1, CPU:0` |
| 编译算子 | 每段 24 个 `QuantBatchMatmulV3`、16 个 `StaticQuant`、8 个 attention `BatchMatMulV2` |
| 图大小 | 每段约 31.99 MB，完整运行包约 3.23 GiB |
| 运行配置 | 固定 seq_len=608，`visual_blocks_om_deepstack_dup=[2]`，App 离线 NPU 图模式 |

配置参考华为[三段式量化说明](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-llm-three-stage-quantification)及[量化效果评估](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-llm-quantization-effect-evaluation)。这是官方插件配置的隔离验证，不代表完整复现 CUDA 三段式优化。

### 13.3 UINT4/LUT 与 ONNX 必须配套

旧 `quant_param_2=False` 分支生成 INT4，当前 Kirin9030 编译器拒绝该组合；SDK 的 True 分支生成 UINT4/LUT。此时同名 `fake_quant_weight.pth` 中被使用的 Linear 权重是浮点类型保存的整数索引 `0..15`，并非可直接前向的反量化权重。

必须先用 LUT 和 group scale 解码核对权重，再将**索引权重**重导到配套 ONNX，传入对应 `quant_params_file`。本次六段全部 24 个 Linear 的 LUT 解码值与原 W4 CPU fake-quant 权重逐值一致；ONNX MatMul initializer 也与配套索引 checkpoint 的转置逐值一致。只换量化参数文件、沿用旧浮点权重 ONNX 会破坏配对。

最终 ONNX initializer 保留 FLOAT32 索引。提前改成 FLOAT16 会被本次 W4 压缩前处理拒绝；OMG 的 `--weight_data_type=FP16` 不代表 `compress_conf` 下的实际权重变成 FP16。此次 `.om` 文件沿用 App 命名，内容实际为 OMC。

这些检查确认本次权重编码配对正确；由于 W4A16 同时改变策略、位宽、分组及编码，不能仅据其真机可用就断定旧 W8A8 故障由哪一项引起。

### 13.4 电脑端精度结论仍保留

八张留出截图的最终视觉特征相对原始浮点参考：cosine 为 **0.860–0.918**，relative L2 为 **39.9%–52.1%**。另测一张猫照片，cosine 为 **0.853**，relative L2 为 **60.0%**。主要误差来自 W4 权重量化；增加 A16 输入/输出量化相对 W4 仅权重量化参考的截图 relative L2 中位数为 **1.37%**、最大为 **16.82%**。

相同浮点 HF LLM 的 CPU 解码测试中，三张界面截图及一张猫照片的 W4A16 回答前缀均与图片相关；部分回答在 80 tokens 截断。这是小样本图文检查，不等同于手机上的 W8 LLM，也不是完整准确率基准。

此前 W8A8 加 INT16 输出候选也测过 CPU relative L2。在共同样本 0、1 的各段独立输入记录中，W4A16 有两段误差更小、四段更大；校准配置和参考前向并非完全统一，不能据此做严格性能排名，但已有结果不支持“W4A16 整体精度优于 W8A8”。此次真机反馈改变的是本发布包的图文可用状态，没有改变上述数值结论。

### 13.5 使用与后续复核

现有 App 不需要更新或重新安装。先清空旧的 ModelScope 仓库输入框，填写上述仓库，正常下载后选中新模型，切换“离线 NPU 图”并重新加载；不要用旧 MNN block 的 CPU 路径评估本次离线图。

后续如收集运行日志，仍应检查 `NPU_ACTIVE`、`graph_ready=6`、`cpu_fallback=0`、`npu_error=0`。这些是待核验项，不是此次用户反馈中已提供的日志证据。固定输入 probe 的严格诊断阈值不替代用户的图文验收标准。

上述发布包的隔离构建与主机报告位于 `/tmp/vit_w4a16_g64_official/`，发布文件及哈希审计位于 `/tmp/w4a16_modelscope_publish/`；临时目录仅作本次追溯，长期产物以上述 ModelScope 仓库为准。后续已将配套 UINT4/LUT 导出接入正式入口，并新增 [`build_kirin_offline.py`](../../transformers/llm/export/plugin_quant_visual_matmul_route_v1/build_kirin_offline.py) 覆盖完整流程，使用方法见 [README §2.6](../../README.md#26-kirin9030-完整离线编译vit-w4a16--llm-cpu)。新入口生成的包仍需独立真机验证；手机 App 未修改。

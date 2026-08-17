# 改动说明 —— 工作树变更汇总 (2026-08-15)

> 本文档汇总当前工作树中所有已修改和新增的文件，分析每项更改的作用、彼此的关联性，以及潜在的冗余/矛盾。

---

## 目录

1. [硬件/后端层：HiAI NPU Backend](#1-硬件后端层hiai-npu-backend)
2. [Express 运行时层：KVCache 元数据管道](#2-express-运行时层kvcache-元数据管道)
3. [LLM 引擎层：Omni 多模态 + NPU Chunk 执行器](#3-llm-引擎层omni-多模态--npu-chunk-执行器)
4. [构建配置](#4-构建配置)
5. [NPU 离线编译脚本 + Route 导出器](#5-npu-离线编译脚本--route-导出器)
6. [校准 Manifest 清理](#6-校准-manifest-清理)
7. [文档](#7-文档)
8. [Skill 更新](#8-skill-更新)
9. [跨改动关注点分析](#9-跨改动关注点分析)

---

## 1. 硬件/后端层：HiAI NPU Backend

### 1.1 原子写 + 防撕裂写 OM 文件

**文件**: `source/backend/hiai/backend/NPUBackend.cpp` — `WriteToOMFile()`

- 写入策略从 `fopen(目标) → fwrite → fclose` 改为 `fopen(临时.tmp) → fwrite → fflush → fsync → fclose → rename → remove(临时)`。
- **作用**：防止进程崩溃或掉电时留下半写文件被当作合法缓存加载。
- **与 CMAKE 无关**：纯防御性改进，不影响接口。

### 1.2 资源释放全面性

**文件**: `NPUBackend.cpp` — `onResizeEnd()` / `NPUBackend` 析构处理

- `mModelName.clear()` 在重置时被调用。
- `mMgrClient.reset()` 在 UnLoadModel 后显式释放智能指针，确保资源及时归还。
- `resetLoadedTensorInfo()` 新增函数：集中清理 `mInputDimension`、`mOutputDimension`、`mInputTensors`、`mOutputTensors`、`mMNNOutTensors`、`mOutputMemToIndex`，并条件性清理 `mInputMap`。
- **作用**：消除因多次 load/reload 未正确重置中间状态导致的陈旧引用或哈希碰撞。
- **冗余/矛盾**：`resetLoadedTensorInfo()` 的部分清除项在原有代码个别位置可能已被直接赋值清理。但集中到一个函数是好的重构。

### 1.3 Shape 感知的 OM 缓存键 (MNN_HIAI_CACHE_OM_BY_CHUNK)

**文件**: `NPUBackend.cpp` — `buildOmCacheShapeKey()` (条件编译)

- 新的缓存路径：`<npu_model_dir>/om_cache_v2/<64位FNV-1a哈希>/vision.om`
- 哈希包含：graph 中 op 数量、chunk 名、每个输入/输出 tensor 的维度、数据类型、format。
- **作用**：同一 chunk_dir 下不同输入 shape 的 OM 不再互斥，多个 shape 缓存共存。
- **潜在问题**：哈希依赖 `mInputOrder`，但 `mInputOrder` 在 `getInOutTensorInfo()` 中才被填充，而 `buildOmCacheShapeKey()` 在其之前调用。若 `MNN_HIAI_USE_LOCAL_NPU_FIXES` 启用且后端实例被复用，首次编译和第二次加载时的 key 可能不一致。在实际 LLM 推理场景中，每个 NPUBackend 仅调用 `bulidIRModelAndLoad` 一次，且都以空 `mInputOrder` 起始，因此 key 在多次运行时一致。**但对输入 shape 的动态区分不产生作用**，因为空 `mInputOrder` 下输入 tensor 信息不会被编入 key。

### 1.4 缓存验证 + 失败自动回退

**文件**: `NPUBackend.cpp` — `bulidIRModelAndLoad()`

- 缓存命中后不再无条件返回：加载后执行 `getInOutTensorInfo()`，若 I/O 映射校验失败或输出匹配落空，则 UnLoadModel、reset、删除破损缓存文件、重新编译。
- 新编译成功后不再立即写缓存，而是等 `getInOutTensorInfo()` 验证通过后才写。
- 新增 `outputMappingValid` 检查：当所有 MNN output 都无法匹配 DDK output 槽位时返回 -1。
- **作用**：使缓存层具备自愈能力，避免"缓存存在但 I/O 排列已变"时静默返回错误。
- **冗余/矛盾**：无。这是对之前 "缓存命中即 OK" 策略的升级。

### 1.5 Process 入参防御

**文件**: `NPUBackend.cpp` — `process()`

- 新增 `mMgrClient` 为 null、`modelIndex` 超出 `mModelName` 范围、`mModelName[modelIndex]` 为空的防御检查。
- `process()` 中原来用 `to_string(modelIndex)` 作为模型名传给 HiAI 的 `context.AddPara`，改为使用实际注册的 `modelName`。
- **作用**：避免空指针崩溃；修正了可能传整数索引而非模型名的潜在错误。`modelName` 在成功加载后才 push 入 `mModelName`，所以不会出现空串。

### 1.6 头文件更新

**文件**: `source/backend/hiai/backend/NPUBackend.hpp`

- `resetLoadedTensorInfo()` 和 `buildOmCacheShapeKey()` 声明。
- 注释更新：缓存路径描述改为 `chunk_i/om_cache_v2/<shape-key>/vision.om`。
- 所有公开/私有接口签名未变。

### 1.7 冗余项：`0001-support-chunk-cache.patch`

- **这是一个 git format-patch**，内容与 `NPUBackend.cpp`、`NPUBackend.hpp`、`build_64.sh` 的当前改动高度重叠（也包括 `CMakeLists.txt` 中 option 描述的文本修改，但工作树未包含该部分）。
- **问题**：若与工作树的改动一并提交，会引入冲突和重复变更。该 patch 应在代码稳定后清理删除。

---

## 2. Express 运行时层：KVCache 元数据管道

### 2.1 问题背景

原 `Executor::RuntimeManager::setHintPtr(Interpreter::KVCACHE_INFO, value)` 直接迭代 `mRuntime.first` 将所有 runtime 的 `pMeta` 设为目标值。但 LLM 中 RuntimeManager 被复用/克隆，而 runtime 实例在运行时可能被共享（池化），此时直接修改 runtime 上的 `pMeta` 不保证新创建的 backend 能继承该值。

### 2.2 改动方案

| 文件 | 改动 |
|------|------|
| `include/MNN/expr/Executor.hpp` | 新增 `applyMetaToRuntime()` 公开方法 |
| `express/RuntimeAttr.hpp` | 新增 `void* mMeta = nullptr` 字段 |
| `express/Executor.cpp` | `setHintPtr(KVCACHE_INFO)` 仅存到 `mInside->mMeta`，不再直接修改 runtime |
| `express/module/Module.cpp` | clone 时拷贝 `mMeta` |
| `express/module/PipelineModule.cpp` | `load()` 调用 `applyMetaToRuntime()` |
| `express/module/StaticModule.cpp` | 构造时、`onForward()` 时、`clone()` 时都调用 `applyMetaToRuntime()` |

### 2.3 时序推理

- `setHintPtr(KVCACHE_INFO, ptr)` → 存储到 `mMeta`
- `applyMetaToRuntime()` → 遍历当前所有 runtime 设置 `pMeta = mMeta`
- 调用点：PipelineModule::load（仅在加载时）、StaticModule 构造函数、`StaticModule::onForward()` 每次推理前

### 2.4 潜在矛盾 / 注意点

1. **冗余调用**：`StaticModule::onForward()` 每次前向都调用 `applyMetaToRuntime()`，如果 KVCACHE_INFO 只在 prefill 阶段设置而 decode 阶段不变，则该调用是安全的但多余。但如果 KVCache 信息确实需要在每次 decode 前更新（追加已缓存 token 数），则这是正确做法。

2. **可见性**：`setHintPtr` 现在只处理 `KVCACHE_INFO` mode，其他 mode 走旧路径直接写 runtime。这是有意地分治，但引入分支意味着未来添加新的 hint mode 需要明确选择分支。

3. **Module::clone 中的拷贝**：`const_cast` 修改 `mMeta`，与原有 `const_cast` 修改 `mContent` 风格一致，但在 const 正确性上并不优雅。

---

## 3. LLM 引擎层：Omni 多模态 + NPU Chunk 执行器

### 3.1 Chunk 执行器接口扩展

**文件**: `transformers/llm/engine/include/llm/npu_chunk_executor.hpp`

- 新增纯虚函数 `virtual size_t chunkSequenceLength(int chunkIdx) const = 0`。
- **作用**：让 Omni 在运行 OM chunk 前能验证 attention_mask 尺寸是否匹配 chunk 的固定输入 shape。

### 3.2 setNpuChunkExecutor 返回值

**文件**: `llm.hpp`, `omni.hpp`

- `setNpuChunkExecutor` 返回类型从 `void` → `bool`，Omni 覆写实现返回 `mNpuChunkExecutor != nullptr`。
- **作用**：调用方（App 层）可通过返回值确认注入是否成功，不再需要单独判空。

### 3.3 OM 加载策略收紧

**文件**: `omni.cpp` — `load()`

- 旧：遍历所有 chunk，仅当 `chunkRunOnNpu[i] && !mNpuChunkOmPaths[i].empty()` 时加载 OM，且不要求所有 NPU chunk 都有 OM 路径。
- 新：遍历所有 chunk，当 `chunkRunOnNpu[i]` 为 true 时，要求 `mNpuChunkOmPaths[i]` 非空，否则 `return false`。
- **作用**：离线 OM 模式下，所有标记为 NPU 的 chunk 必须提供预编译 OM 路径，不允许部分 fallback 到 MNN。
- **语义变化**：这是一个 breaking change —— 任何标记为 NPU 但未提供 OM 路径的 chunk 会导致 load 失败。这要求 App 配置与模型 config.json 严格对齐。

### 3.4 Visual KV 提示默认关闭

**文件**: `omni.cpp` — `load()`

- 旧：`visual_blocks_kv_hints` 默认为 `true`
- 新：`visual_blocks_kv_hints` 默认为 `false`
- **作用**：Visual self-attention 不应消费 decoder KVCache 元数据。显式 opt-in 仅用于实验性模型。
- **潜在破坏**：依赖该默认值为 true 的旧 App 需要显式设置 `"visual_blocks_kv_hints": true`。

### 3.5 OM Chunk 输入验证 + 批量维保持

**文件**: `omni.cpp` — `qwen2VisionProcess()`

- 在 `runChunk` 前检查 `maskVec.size()` 是否等于 `fixedSequenceLength²`。
- 从 `omOutputs[0]` 构造 VARP 时，若 `dims.size() == 2`（来自 visual_pre 或上一个 MNN CPU chunk 的输出），则在头部插入批量维 1，匹配 OM 的 `[1, S, D]` 输出格式。
- **作用**：防止 shape 不匹配导致的静默错误；保证 OM pipeline 和 MNN pipeline 的输出 shape 一致。

### 3.6 解码阶段 MRoPE 位置修复

**文件**: `omni.cpp` — `tokenizer_encode()` 和 `gen_position_ids()`

- 新增 `mDecodePositionBase` 成员，在 `tokenizer_encode()` 时计算：
  `mDecodePositionBase = mContext->all_seq_len + (ids.empty() ? -1 : mPositionIds.back())`
- `gen_position_ids()` 中 decode 阶段的 base 从 `mContext->gen_seq_len + mPositionIds.back()` 改为 `mDecodePositionBase + mContext->gen_seq_len`。
- **作用**：tokenizer_encode 执行在 prefill 推进 `all_seq_len` 之前，是唯一明确的时机来将保留的 prefix KV 偏移与当前回复的压缩多模态位置 timeline 对齐。空 prompt 时保留 prefix 端点不变。
- **注意**：`ids.empty() ? -1` 这个边界条件没有显式注释为什么减去 1。从逻辑推断：空 ids 发生在 prefix cache 的续写场景，此时 `mPositionIds.back()` 不可用（未填充），加 -1 使 base 落在 `all_seq_len - 1` 以避免 decode 阶段的 `gen_position_ids` 产生跳跃。

### 3.7 Omni 头文件新增

**文件**: `omni.hpp`

- `setNpuChunkExecutor` 返回 `bool` 并加 `override`。
- `mDecodePositionBase: int` 成员。
- 无 API 破坏，但 `mContext->all_seq_len` 的公开程度需要 `Context` 结构中 `all_seq_len` 可见。

---

## 4. 构建配置

### 4.1 HarmonyOS 构建

**文件**: `project/harmony/build_64.sh`

- `MNN_HIAI_CACHE_OM_BY_CHUNK=OFF` → `ON`
- **作用**：在鸿蒙设备上启用新的 shape 感知 OM 缓存路径。
- **影响**：所有使用该脚本的 HarmonyOS 应用将自动使用新缓存机制。

### 4.2 CMake 变更待确认

`0001-support-chunk-cache.patch` 中包含了 `CMakeLists.txt` 的更改（option 描述文本从 `"Enable chunk-aware OM cache path for HiAI backend"` 改为 `"Enable chunk- and shape-aware OM cache for HiAI backend"`），**但该更改尚未被 `git diff` 反映**，说明 patch 尚未 apply 到工作树，或已通过其他方式包含。当前工作 tree 的 CMakeLists.txt 与此 patch 有冲突风险。

---

## 5. NPU 离线编译脚本 + Route 导出器

### 5.1 visual_plugin_quant_matmul_route.py

| 变更 | 说明 |
|------|------|
| DOPT 导入变成可选（try/except） | FP16 Route 不需要 DOPT 依赖 |
| 新增 `require_dopt()` | 在依赖 DOPT 的命令入口显式报错 |
| 新增 `export_raw_fp16_onnx()` | 从原始 HF 浮点权重直接导出 FP16 ONNX，不经过 fake-quant weights |
| 新增 `export_chunk_onnx()` | 抽取公用 ONNX 导出逻辑，接受 `state` 和 `weight_source` 参数 |
| 旧 `export_onnx()` 重写为调用 `export_chunk_onnx` | 向后兼容，仍使用 fake_quant_weight.pth |
| 新增 cmd `"export-fp16"` | 通过 `--fp16 --sequence_length --hidden_size --rotary_size` 参数控制 |
| export report 新增 `weight_source` 和 `calibration_used` | 便于审核产物的权重来源 |

- **作用**：分离两条导出路径——旧 DOPT W8A8 校准路径和新原始 HF FP16 路径。FP16 导出不需要校准图片、NPZ 或 fake_quant_weight.pth。
- **冗余/矛盾**：无。两者在同一个脚本中共存，通过 `cmd` 参数选择，互不干扰。

### 5.2 run_visual_plugin_matmul_omc.sh

| 变更 | 说明 |
|------|------|
| 新增 `TARGET_MODEL_TYPE` 控制 | 取值 `"om"`（在线 IR）或 `"omc"`（离线编译），默认 `"om"` 保持兼容 |
| 新增 `USE_COMPRESS_CONF` 控制 | 默认 `true`，设为 `false` 即非压缩 FP16 |
| 新增 `LOAD_ASCENDC_ENV` 和自动检测 | Kirin9030 + OMC 时自动 source AscendC 环境，导入后验证 `te_fusion` |
| 新增输出目录创建 | `mkdir -p "$(dirname "${OUTPUT_PREFIX}")"` |
| `OUTPUT_PREFIX` 可覆盖 | 默认值不变，但允许调用方指定 |
| compress_conf 条件性加入 omg_args | 只有 `USE_COMPRESS_CONF=true` 时才传 `--compress_conf` |
| 产物存在性检查 | OMG 完成后检查 `OUTPUT_FILE` 是否存在且非空 |
| `set +e/+u` 保护 AscendC 脚本导入 | Huawei 脚本可能包含未定义变量，需要临时放宽 shell 选项 |

- **作用**：通用化 OMG 包装脚本，同时支持在线 IR (om) 和离线 OMC 编译，支持 Kirin9030 的 AscendC 环境加载。
- **注意**：`ASCENDC_ENV_SCRIPT` 默认路径推导依赖 `${OMG_TOOL}` 的目录结构，与 DDK 实际布局耦合。

### 5.3 run_all_chunks_real_calib_W8A8.sh

- `PLATFORM=kirin9020` → `PLATFORM=kirin9030`
- **作用**：旧 W8A8 校准脚本从 Kirin9020 切换为 Kirin9030 目标平台。
- **矛盾**：该脚本沿用 DOPT W8A8 校准路径，而在 Kirin9030 的 FP16 OMC 路线上 W8A8 压缩已被 `USE_COMPRESS_CONF=false` 禁用。除非此脚本另有用途（比如为 CPU/视觉 fallback chunk 保留 W8A8），否则它与主流程的 FP16-only 策略不一致。需要确认该脚本在最终流水线中是否仍被调用。

### 5.4 新增：build_kirin9030_offline_model_from_gptq.sh (untracked)

- 端到端构建入口，涵盖：模型一致性检查 → CPU/MNN W8G128 导出 → HF FP16 Route ONNX → Kirin9030 OMC 编译 → 验证 → 带回滚保护的目录激活。
- 可接受环境变量或位置参数。

### 5.5 新增：compile_kirin9030_offline_fp16_chunks.sh (untracked)

- 简化版：已有完整 MNN 模型时，仅对 NPU chunk 重新编译 OMC。
- 包含 staging 目录、平台插件校验（SHA-256）、回滚逻辑、目录内激活（避免 `/temp` 不支持目录 rename）。

---

## 6. 校准 Manifest 清理

**文件**: `transformers/llm/export/plugin_quant_visual_matmul_route_v1/calib_inputs_real/visual_calib_manifest.json`

- 删除了每个 chunk 中 `sample_idx=1` 的条目（保留 `sample_idx=0`），以及所有 `dtypes` 和 `stats` 字段。
- **影响**：清单从 12 条缩至 6 条，仅保留 shape 元数据。
- **矛盾**：如果 `run_all_chunks_real_calib_W8A8.sh` 仍使用 `NUM_SAMPLES=2`，会在加载第 2 个样本时找不到对应 manifest 条目。需要确认调用方是否仍需要 2 样本。

---

## 7. 文档

**文件**: `CLAUDE.md`

- 新增了 `OMG/OMC 离线 NPU 图编译（Kirin9030）` 和 `环境配置` 两个大节，详细记录了：
  - om vs omc 产物区别
  - DDK / AscendC 环境准备
  - 端到端入口脚本用法
  - 成功检查标准与发布要求
  - 单 chunk 调试示例
- **作用**：确保后续维护者清楚 Kirin9030 离线编译的完整流程。

---

## 8. Skill 更新

**文件**: `skills/retrospective/SKILL.md`

- 新增 "跨机器补丁安全" 和 "厂商工具链与离线产物校验" 两个反思记录。
- 记录了 LF/BOM 规范化、blob 哈希比较、日志语义验证而非仅看 success 字样、FP16 权重来源审计等 lessons learned。
- **作用**：驱动 skill 持续进化，防止同类问题重犯。

---

## 9. 跨改动关注点分析

### 9.1 功能耦合度

| 改动域 | 依赖关系 |
|--------|---------|
| NPU Backend (C++) | 独立，仅使用条件编译宏 `MNN_HIAI_CACHE_OM_BY_CHUNK` |
| Express Runtime | 独立，影响 KVCACHE hint 传播 |
| Omni LLM Engine | 依赖 `INpuChunkExecutor` 接口和 `chunkSequenceLength` |
| 导出脚本 (Python) | 独立，供 build 流水线调用 |
| Shell 脚本 (OMC) | 独立，Python + OMG 工具 |

- 没有循环依赖。C++ 引擎层的改动（Express + NPU Backend + Omni）是垂直的，导出脚本链是水平的。

### 9.2 冗余项

1. **`0001-support-chunk-cache.patch`** — 与当前 `NPUBackend.cpp`/`.hpp`/`build_64.sh` 的改动高度重叠，提交前应删除。
2. **`StaticModule::onForward()` 中每次调用 `applyMetaToRuntime()`** — 在 KVCACHE_INFO 不变化的 decode 阶段属于安全但多余的执行。当前逻辑下每步 decode 都遍历一次所有 runtime 设 `pMeta`，性能影响小但值得备注。

### 9.3 矛盾 / 风险项

1. **`visual_calib_manifest.json` 精简 + `run_all_chunks_real_calib_W8A8.sh` 的 `NUM_SAMPLES=2`**：若该脚本仍需要 2 个样本的 manifest 数据，会因条目缺失而失败。应确认该脚本是否已从主流程下线。

2. **`run_all_chunks_real_calib_W8A8.sh` 的 `PLATFORM=kirin9030`**：该脚本走 DOPT W8A8 + compress_conf 路径，与 Kirin9030 的 FP16 OMC 策略矛盾。除非有明确理由（例如为 CPU fallback chunk 生成 W8A8 route），否则需要确认。

3. **OM Chunk 输入验证的固定 shape 假设**：`chunkSequenceLength(i)` 在 `qwen2VisionProcess` 中被假设为返回非零值，且 mask 验证要求 `size() == fixedSequenceLength²`。如果某个 OM chunk 被加载为不同 sequence length，会拒绝输入。**但**由于当前模型固定 S=608，此假设在实践中成立。若未来支持动态 S，需放宽该检查。

4. **`buildOmCacheShapeKey()` 的时序依赖**：当 `MNN_HIAI_USE_LOCAL_NPU_FIXES` 启用时，shape key 依赖的 `mInputOrder` 只有在 `getInOutTensorInfo()` 后才被填充。后端实例只应一次 `bulidIRModelAndLoad`，因此实际不影响正确性，但其逻辑依赖隐式推理，代码可维护性较低。

### 9.4 整体评估

- **无 Data Race / 并发问题**：所有关键路径都在单线程 backend 初始化中执行。
- **向后兼容**：除了 `visual_blocks_kv_hints` 默认值翻转（section 3.4），所有改动均向后兼容。旧 App 只需显式设置该配置即可恢复原行为。
- **编译**：添加的头文件包含和条件编译宏不会破坏现有构建。`MNN_HIAI_CACHE_OM_BY_CHUNK` 默认值保持 `OFF`（在 CMakeLists.txt 中），HarmonyOS 构建脚本单独设为 `ON`。
- **测试建议**：需要回归的测试点：(1) 未设置 KVCACHE_INFO 的纯 MNN 模型；(2) 未启用 OM cache 的旧 HiAI 路径；(3) 仅有 CPU chunk 的视觉模型（无 NPU chunk）。

---

*本文档生成于 2026-08-15，仅基于 `git diff` 和新增文件分析，未修改或运行任何代码。*

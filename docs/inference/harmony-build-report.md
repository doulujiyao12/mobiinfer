# 鸿蒙 NPU 构建报告（libMNN.so + 8-bit 量化模型导出）

- 日期：2026-08-08
- 目的：按 `README.md` §2（Kirin NPU 用法）生成鸿蒙端 `libMNN.so` 与适配 Kirin NPU 的 8-bit 量化模型（用户的 GPTQ 为 **8-bit** 量化，README 示例为 4-bit，已适配）

---

## 一、lib 产物

| 项 | 值 |
|---|---|
| 位置 | `/home/ma-user/workspace/feh/mobiinfer/project/harmony/build/libMNN.so` |
| 大小 | 11,971,984 字节（≈ 11.4 MiB） |
| 格式 | ELF 64-bit LSB shared object, **ARM aarch64**, with debug_info, not stripped |
| BuildID | 6880005f1944b9ba0046536c4a5dc9fd2048affa |
| 动态依赖 | `libhiai.so`, `libhiai_ir.so`, `libhiai_ir_build.so`, `libhilog_ndk.z.so`, `libc.so` |
| RUNPATH | `/home/ma-user/workspace/feh/mobiinfer/source/backend/hiai/3rdParty/arm64-v8a`（运行时需随 app 打包 HiAI 库，见 §五） |

### 构建配置（CMakeCache）

| 选项 | 值 |
|---|---|
| CMake 工具链 | `/temp/csm/command-line-tools/sdk/default/openharmony/native/build/cmake/ohos.toolchain.cmake`（HARMONY_HOME） |
| `OHOS_ARCH` | `arm64-v8a` |
| `MNN_NPU` | `true`（HiAI 后端） |
| `MNN_BUILD_LLM` | `ON` |
| `MNN_BUILD_LLM_OMNI` | `ON`（多模态，含 `INpuChunkExecutor`） |
| `MNN_LOW_MEMORY` | `ON` |
| `MNN_BUILD_SHARED_LIBS` | `ON` |
| `CMAKE_BUILD_TYPE` | `Release` |

- 构建方式：`cd project/harmony/build && ../build_64.sh`（构建前已删除旧 CMakeCache 做干净配置）
- 编译结果：**0 errors**（会话中记录约 843 条 warnings，均为第三方/非关键告警）
- 编译耗时：约 8 分钟（128 核并行）

---

## 二、模型产物

位置：`/temp/models/mnn_mobi_2B_w8a8_visual_npu/`（共 33 个文件，总大小 **4.7 GB**）

### 文件清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `llm.mnn` + `llm.mnn.weight` | 509 KB + 2.12 GB | 语言模型（28 层，GPTQ **8-bit** w8g128 已应用） |
| `embeddings_bf16.bin` | 622 MB | 独立 embedding（`--seperate_embed`，bf16） |
| `tokenizer.mtok` | 4.1 MB | 分词器 |
| `visual_pre.mnn` | 4.7 MB | 视觉 pre（patch_embed + merger，fp16，权重内联） |
| `visual_blocks_npu_0..5.mnn`（×6） | 各 ≈ 54 MB（.weight） | 视觉 transformer 块，均分为 6 个 chunk，GPTQ 8-bit |
| `visual_post.mnn` | 18 KB + 201.5 MB（.weight） | 视觉 post（deepstack 聚合等，fp16） |
| `config.json` | 1.2 KB | 引擎运行配置（见下） |
| `llm_config.json` / `export_args.json` | — | 模型结构配置 / 导出参数记录 |

> 视觉部分权重总量：pre 3.1 MB + 6×53.7 MB + post 201.5 MB ≈ **526 MB**，与团队此前验证过的 e2e 导出（`mnn_mobi_gptq_new_sym_e2e_2B_w8a8_half_rl_n64_s512_visual/`，visual.mnn.weight = 526 MB）完全一致，只是拆分存储。

### 关键运行配置（config.json）

```json
"visual_split": true,
"visual_blocks_backend_type": "hiai",
"visual_blocks_chunks": ["visual_blocks_npu_0.mnn", ..., "visual_blocks_npu_5.mnn"],
"visual_blocks_chunk_backends": ["npu", "npu", "npu", "npu", "cpu", "cpu"]
```

- 24 个视觉块按层均分为 **6 个 chunk × 4 层**；前 4 个 chunk（第 0–15 层）路由到 **NPU**，后 2 个（第 16–23 层）路由到 **CPU**（deepstack 层 5/11/17 分别落在 chunk 1/2/4，其输出由各 chunk 按全局索引输出）。
- 引擎端（`omni.cpp`）按 `visual_blocks_chunks` 非空走 chunk 加载路径（优先级高于单体 `visual_blocks.mnn`），逐 chunk 做 HiAI IR-build / OM 编译，降低单次编图内存峰值。

### 量化适配说明（8-bit vs README 4-bit）

| 参数 | README（4-bit） | 本次（8-bit，按用户 GPTQ） |
|---|---|---|
| `--quant_bit` / `--quant_block` | 4 / 128 | **8 / 128** |
| `--visual_quant_bit` / `--visual_quant_block` | 4 / 128 | **8 / 128** |
| `--gptq_path` | — | `/temp/csm/autoround_export/mobi0402_2B_halfimage_rl-w8g128/` |
| `--visual_gptq_path` | — | 同上（与上次成功导出一致） |
| `--lm_quant_bit` | 16 | 16 |

GPTQ 权重来源确认：`quantization_config.json` → `bits=8, group_size=128, sym=true`，量化范围 `model.visual.blocks, model.language_model.layers`，格式 `auto-round:auto_gptq`。

导出日志确认：`Visual GPTQ: replaced 24/24 block weights`（全部视觉块替换为 int8，merger/deepstack/patch_embed 保持 fp16）；语言模型 `apply gptq to llm.mnn.weight` 完成（2.12 GB，符合 w8a8 预期）。

---

## 三、导出命令（8-bit 适配）

```bash
cd transformers/llm/export
python llmexport.py \
  --path /temp/models/mobi0402_2B_halfimage_rl \
  --export mnn \
  --gptq_path /temp/csm/autoround_export/mobi0402_2B_halfimage_rl-w8g128/ \
  --visual_gptq_path /temp/csm/autoround_export/mobi0402_2B_halfimage_rl-w8g128/ \
  --quant_bit 8 --quant_block 128 \
  --visual_quant_bit 8 --visual_quant_block 128 \
  --lm_quant_bit 16 \
  --seperate_embed \
  --visual_split --visual_npu_chunks 6 \
  --visual_chunk_backends "npu,npu,npu,npu,cpu,cpu" \
  --mnnconvert /home/ma-user/workspace/feh/mobiinfer/build-host-converter/MNNConvert \
  --dst_path /temp/models/mnn_mobi_2B_w8a8_visual_npu
```

> 注：README 中 `--visual_npu_chunk 6` 为单数，代码 parser 实际为 **`--visual_npu_chunks`**（复数），本次使用正确写法。

### 导出环境

- Python：`/home/ma-user/.conda/envs/cann/bin/python`（3.10.20）
- 依赖：torch 2.6.0+cu124、transformers 5.9.0、onnx 1.22、onnxruntime 1.23、safetensors；本次补充安装 `yaspin`（ENV.md 允许在 CANN 环境添加依赖）
- `MNNConvert`（host 转换器）：为匹配仓库导出逻辑，由本仓库源码编译 → `build-host-converter/MNNConvert`（13.4 MB，`-DMNN_BUILD_CONVERTER=ON`）

---

## 四、验证结果

| 检查项 | 结果 |
|---|---|
| libMNN.so 为 OHOS arm64-v8a ELF、链接 HiAI 三库 | ✅ |
| CMake 配置（MNN_NPU/LLM/OMNI/LOW_MEMORY） | ✅ |
| 编译 0 errors | ✅ |
| 导出全程 0 Traceback / 0 Error（日志 grep） | ✅ |
| 视觉 6 chunk 全部 GPTQ 8-bit 替换（24/24） | ✅ |
| LLM 侧 GPTQ 应用完成（llm.mnn.weight 2.12 GB） | ✅ |
| 每个 `.mnn` + `.weight` 运行时加载（MNNConvert 转 json 回读，pre/0/5/post 均成功） | ✅ |
| 权重总量与已 e2e 验证的旧导出一致（526 MB） | ✅ |
| config.json chunk 路由（6 文件 + cpu/npu 后端数组）与 omni.cpp 期望格式一致 | ✅ |

> 未做：真机（Kirin 设备）端到端推理验证——需在设备上执行 OM 编译与 NPU 运行，见 §五。

---

## 五、部署到鸿蒙 App 的下一步

1. **替换 lib**：将 `libMNN.so` 复制到 app 的 lib 目录，同时按 README §2.1 打包 HiAI 运行时库（`ddk/ai_ddk_lib/lib64/*`：libhiai.so、libhiai_ir.so、libhiai_ir_build.so）。
2. **放置模型**：将 `mnn_mobi_2B_w8a8_visual_npu/` 整个目录作为模型目录（`config.json` 同级）。
3. **OM 编译（设备端）**：首次运行时 HiAI 后端会对 4 个 NPU chunk 做在线 IR-build；如需缓存，设置 `MNN_HIAI_CACHE_OM_BY_CHUNK`，OM 文件输出到 `chunk_i/vision.om`（详见 `docs/inference/harmonyos-npu.md` §5.4–5.5）。
4. CPU chunk（chunk 4/5）由 MNN CPU 后端执行，无需额外配置。

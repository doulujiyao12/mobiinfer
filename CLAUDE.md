# MNN Project Instructions

MNN is a lightweight deep learning **inference engine** (not a training framework), targeting mobile and server platforms. Supports CNN / Transformer / LLM / Diffusion models. Code must prioritize **performance and binary size**.

## Restricted Access

> The following directories contain internal proprietary code. **Do NOT read, modify, or reference** any files within:
> - `schema/private/`
> - `source/internal/`

## Architecture Overview

MNN uses a **graph optimization + heterogeneous backend scheduling** architecture.

Two inference APIs are available:
- **Session API** (low-level): `Interpreter → createSession → runSession`, operates on Tensor directly
- **Module API** (high-level, recommended): `Module::load → onForward(VARP)`, Express-based dynamic graph. Used by LLM / Diffusion and most modern workloads

**Key abstractions** (see corresponding headers under `source/core/`):
- **Interpreter** / **Session**: model loading and inference session management
- **Backend** / **Execution**: hardware backend abstraction and per-op implementation (CPU/Metal/CUDA/OpenCL/Vulkan/...)
- **Tensor**: data container; internally uses NC4HW4 format (channels packed by 4 for SIMD)
- **Op / Schema**: FlatBuffers-defined operator descriptors (`schema/default/*.fbs`)

**Op registration pattern**: Schema definition → shape inference (`source/shape/`) → Geometry decomposition (optional) → Backend Execution implementation

## LLM Subsystem

MNN supports end-to-end LLM export and inference:

- **Python export** (`transformers/llm/export/`): HuggingFace model → MNN format. Core modules: `llmexport.py` entry point, `utils/model_mapper.py` (model field mapping), `utils/model.py` (unified LlmModel class), `utils/transformers.py` (Attention/Decoder/RoPE export)
- **C++ inference** (`transformers/llm/engine/`): `llm.cpp` (text inference), `omni.cpp` (multimodal: vision/audio), includes KVCache management and sampling strategies

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `include/MNN/` | Public C++ headers |
| `source/core/` | Inference core (Interpreter, Session, Pipeline, Backend) |
| `source/backend/` | Hardware backend implementations (cpu, arm82, metal, cuda, opencl, vulkan, ...) |
| `source/shape/` | Shape inference |
| `source/geometry/` | Geometry computation (op decomposition) |
| `express/` | Express API (high-level dynamic graph, VARP) |
| `schema/default/` | FlatBuffers schema (op definitions) |
| `tools/converter/` | Model converter (ONNX/TF/Caffe → MNN) |
| `transformers/llm/` | LLM export (Python) + inference engine (C++) |
| `transformers/diffusion/` | Diffusion model support |
| `pymnn/` | Python bindings |
| `test/` | Test cases |
| `skills/` | AI Agent Skills |

## Coding Style

- **C++**: Google Style variant, see `.clang-format`. 4-space indent, 120-char line width, attached braces. Class names `PascalCase`, functions `camelCase`, member variables `mCamelCase`. RTTI and exceptions disabled (`-fno-rtti -fno-exceptions`). Default standard: C++11.
- **Python**: Standard Python conventions
- **Formatting**: `clang-format -i -style=file <file>`

## Build & Test

```bash
# Build C++ (with LLM)
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)

# Common CMake options: MNN_BUILD_TEST, MNN_BUILD_CONVERTER, MNN_METAL, MNN_OPENCL,
# MNN_VULKAN, MNN_CUDA, MNN_ARM82, MNN_BUILD_QUANTOOLS, MNN_SUPPORT_TRANSFORMER_FUSE
# Full list: see option() declarations at the top of CMakeLists.txt

# Unit tests
cd build && ./run_test.out

# LLM export
cd transformers/llm/export
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path ./MODEL

# LLM test
cd build
./llm_demo /path/to/MODEL/config.json prompt.txt

# LLM benchmark
./llm_bench -m /path/to/MODEL/config.json
```

Test suite includes: unit tests (`run_test.out`), model tests, conversion tests (ONNX/TF/TFLite/Torch), quantization tests, LLM tests, PyMNN tests. See `test.sh` and `test/` directory for details.

## Skills

For the following tasks, **read the Skill entry file first** and execute step by step. Each step must pass its tests before proceeding.

**After completing any skill-driven task, run the Retrospective skill** to reflect on mistakes and update the skill with lessons learned.

| Skill | Entry File | Trigger |
|-------|-----------|---------|
| Support new LLM | `skills/support-new-llm/SKILL.md` | Add / adapt a new LLM model |
| Add new op | `skills/add-new-op/SKILL.md` | Add a new operator |
| ARM CPU optimization | `skills/arm-cpu-optimize/SKILL.md` | Optimize op performance on ARM CPU |
| Retrospective | `skills/retrospective/SKILL.md` | After any non-trivial task: reflect on mistakes, update relevant skills with lessons learned |


## 端侧鸿蒙app代买库
mobiinfra-oh 这个目录下面是端侧推理的deveco 的项目代码库，主要的程序文件在mobiinfra-oh/entry/src/main/cpp目录下面，这里mobiinfra-oh/entry/libs/arm64-v8a/libMNN.so这个so文件是project/harmony/build_oh/libMNN.so拷贝过去的


## OMG/OMC 离线 NPU 图编译

### 目标与产物判定

- `--target=om` 生成的是可供设备侧在线编译的 HiAI IR 容器，不是已经针对设备编译完成的离线图。
- 真正的离线 NPU 图必须使用 `--target=omc`。本项目仍使用 `.om` 文件扩展名，以兼容 App/MNN 的 chunk 文件解析逻辑，但必须以 OMG 日志确认其内容来自 OMC 编译。
- **Kirin9030（V311）**：编译必须加载 DDK 自带的 AscendC 环境。只设置平台参数但未加载 AscendC 时，MatMul 可能全部被拒绝并回退到 CPU；即使生成了文件，也不能视为成功。FP16 Route 必须直接从原始 HuggingFace 浮点权重导出，不能先加载 DOPT `fake_quant_weight.pth` 再转成 FP16；后者保存的是伪量化/反量化后的权重，会改变数值和模型精度。
- **Kirin9030（V311）W8A8 已可用（DDK 6.1.1.0）**：见下方「Kirin9030 W8A8 OMC（DDK 6.1.1.0）」专节。旧 DDK 6.0.1.0 的 W8A8 `compress_conf` 路径会令 `MatMulV2` 按 FP16 权重大小校验，而压缩 INT8 buffer 只有预期大小的一半，报 `Size check failed. realSrcSize < expectSrcSize` 后 `Trans weight failed` 回退失败；该问题在 DDK 6.1.1.0 + `kirin9030-plugin-next-6.1.1.0` 中已修复。
- **Kirin9020（V300）**：支持离线 OMC + 旧 DOPT W8A8 权重 + `compress_conf` 的编译路径。该平台上 W8A8 的 MatMulV2 没有 FP16 权重大小校验，压缩后的 INT8 buffer 可以正常编译。编译不需要加载 AscendC 环境（Kirin9020 平台插件自含 tiling 能力）。产出物仍然是 `--target=omc` 生成的离线图，与旧 `--target=om` 在线 IR 有本质区别。**注意**：该产物中的权重来自 DOPT 伪量化压缩，精度路径与 Kirin9030 FP16 OMC 不同，请按目标芯片分别验证。

### 环境准备

先阅读仓库根目录的 `ENV.md`，使用其中记录的当前 DDK/CANN/Conda 路径。不要把 `ENV.md` 中的令牌或密码输出到日志、提交或命令行参数。

> **路径已迁移（2026-10 重启后）**：旧 `/temp/fdh/baiducloud/...` 与 `/temp/models/csm/` 均已不存在，DDK/CANN 现统一放在 `/temp/huawei-sdk/`。下面命令中的路径如与 `ENV.md` 不一致，以 `ENV.md` 为准。

当前 Kirin9030 工具链的关键组成是：

```bash
export DDK_PATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0
source "$DDK_PATH/tools/tools_ascendc/set_ascendc_env.sh"
```

conda 环境名是**大写 `CANN`**（不是小写 `cann`）：

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate CANN
```

如果 DDK 的 Python package 目录缺少适配器，使用 DDK 自带 wheel 离线安装，不要从公网拉取不匹配版本：

```bash
python -m pip install --no-index --no-deps \
  --target "$DDK_PATH/tools/tools_ascendc/package/python" \
  "$DDK_PATH/tools/tools_ascendc/package/ascendc_adapter-0.1-py3-none-any.whl"
```

专用脚本会检测并完成上述 AscendC 配置，通常无需手动执行。


### 单 chunk / 调试编译

通用包装脚本位于：

```text
transformers/llm/export/plugin_quant_visual_matmul_route_v1/run_visual_plugin_matmul_omc.sh
```

Kirin9030 离线 OMC 单 route 示例（FP16、无压缩）：

```bash
PLATFORM=kirin9030 \
TARGET_MODEL_TYPE=omc \
USE_COMPRESS_CONF=false \
bash transformers/llm/export/plugin_quant_visual_matmul_route_v1/run_visual_plugin_matmul_omc.sh \
  /path/to/visual_chunk_route fp16
```

Kirin9020 离线 OMC 单 route 示例（DOPT W8A8、带 compress_conf）：

```bash
PLATFORM=kirin9020 \
TARGET_MODEL_TYPE=omc \
USE_COMPRESS_CONF=true \
bash transformers/llm/export/plugin_quant_visual_matmul_route_v1/run_visual_plugin_matmul_omc.sh \
  /path/to/visual_chunk_route fp16
```

旧在线 IR 路径示例（不要与离线 OMC 混淆）：

```bash
PLATFORM=kirin9020 \
TARGET_MODEL_TYPE=om \
USE_COMPRESS_CONF=true \
bash transformers/llm/export/plugin_quant_visual_matmul_route_v1/run_visual_plugin_matmul_omc.sh \
  /path/to/visual_chunk_route fp16
```

### 成功检查与发布要求

每个离线 NPU chunk 的日志至少应同时包含以下语义；只看到输出文件存在不够：

```text
partition type NPU:1, CPU:0
MemoryCalculateForGraph
SaveCompiledModelToFile SUCCESS
OMG generate offline model success
```

同时确认：

1. 没有 `CPU fallback`、分区为 `NPU:0, CPU:1` 或 MatMul 全部 unsupported；
2. `offline_om_manifest.json` 中的平台、输入 shape、chunk 文件大小和 SHA-256 与最终目录一致；
3. `config.json` 中 NPU chunk 的加载名与执行名一致，NPU/CPU backend 列表正确；
4. 发布或上传 ModelScope 时同步完整运行时精简目录，而不是只上传 `visual_blocks_npu_*.om`；运行时包不包含 `onnx/` 和 `offline_om_build_logs/`，上传后应按路径、大小和 SHA-256 逐文件核对并确认远端孤立中间文件已删除；
5. 仅重新生成 OM/OMC 产物、未修改 MNN 引擎代码时，不需要重新编译或替换 `libMNN.so`；引擎有改动时才重新构建，并在复制到 App 前后校验哈希。

当前固定输入 shape 只覆盖脚本声明的视觉尺寸。若将来改变图片预处理得到的 token 数或模型 hidden size，需要按新 shape 重新编译对应的离线图，不能直接复用旧 OMC 文件。

### Kirin9030 W8A8 OMC（DDK 6.1.1.0）

Kirin9030 的 W8A8 `compress_conf` 路径在旧 DDK 6.0.1.0 下会失败（`MatMulV2` 按 FP16 权重大小校验，压缩 INT8 buffer 只有预期一半，报 `Size check failed` → `Trans weight failed`）。升级到 **DDK-tools-next-6.1.1.0 + kirin9030-plugin-next-6.1.1.0** 后该问题已修复，W8A8 OMC 可正常编译。

关键组成（详见 `ENV.md`）：

```bash
export DDK_PATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0
source "$DDK_PATH/tools/tools_ascendc/set_ascendc_env.sh"   # Kirin9030 必须加载 AscendC
```

首次使用新 DDK 前，需要执行一次 `install.sh`（在 `bash -c` 中 source，zsh 下直接 `source` 会因 `$0` 检测失效）：

```bash
cd "$DDK_PATH/tools/tools_ascendc" && bash -c 'source install.sh'
```

`omg` 与 `omg master` 可执行文件在解压后可能没有执行位，若报 `Permission denied` 需先 `chmod +x`：

```bash
chmod +x "$DDK_PATH"/tools/tools_omg/omg "$DDK_PATH"/tools/tools_omg/master/omg
```

Kirin9030 W8A8 离线 OMC 单 route 示例（DOPT W8A8、带 compress_conf、加载 AscendC）：

```bash
PLATFORM=kirin9030 \
TARGET_MODEL_TYPE=omc \
USE_COMPRESS_CONF=true \
LOAD_ASCENDC_ENV=auto \
OMG_TOOL=$DDK_PATH/tools/tools_omg/omg \
OMG_MASTER_DIR=$DDK_PATH/tools/tools_omg/master \
ASCENDC_ENV_SCRIPT=$DDK_PATH/tools/tools_ascendc/set_ascendc_env.sh \
bash transformers/llm/export/plugin_quant_visual_matmul_route_v1/run_visual_plugin_matmul_omc.sh \
  /path/to/visual_chunk_route fp16
```

成功判据与通用要求相同（`partition type NPU:1, CPU:0` / `SaveCompiledModelToFile SUCCESS` / `OMG generate offline model success`），并确认 `QuantBatchMatmulV3` 量化 matmul 出现在日志中（否则可能是静默回退 FP16）。已用真实校准输入（608 token、W8A8 act_bit=8）编译 6 个 visual chunk，产物约 52MB/chunk，比 FP16 非压缩（49MB）略大，比 Kirin9020 W8A8（98MB）小。Kirin9030 W8A8 路径仍须加载 AscendC 环境，不能像 Kirin9020 那样省略。

> ⚠️ **真机结论修正（2026-10）：visual chunk 的离线 OM 不要用 `compress_conf`（W8A8 激活量化），改用纯 FP16。**
>
> 症状：W8A8 `compress_conf` 编译出的 6 个 `.om` 在真机上**编译与执行都成功**（`RunSync` 全部返回成功、输入元素数匹配、无 NaN/Inf、无 CPU fallback），但**激活数值完全错误**——chunk0 输入正确（absmax≈32.65，与校准 32.28~33.13 吻合），而 chunk0 输出（=chunk1 输入）absmax 达到 **152.75**，校准数据同位置只有 **18.89~24.02**（约 7.5 倍）。表现为回答与图片无关、换图后回答仍相似。
>
> 定位证据：
> 1. **受控对照**：旧仓库（真机验证离线 OM 正常）用 `FP16 + compression:none`；出问题的版本**唯一差异**就是加了 `compress_conf`。而"在线 OM 正常、离线 OM 错误"的分界也正好是 `compress_conf`（在线编译不经过 DOPT 激活量化）。
> 2. **ONNX 无罪**：同一份固定输入下，用于编译的 ONNX（onnxruntime 执行）与 torch 参考前向一致 —— chunk0 输出 absmax 均为 17.28，cos=0.99977（残差仅 fp16 舍入）。6 个 chunk × 4 张图全部 cos=1.0000。所以**偏差不是导出造成的，而是在 OMG 编译/OM 执行环节**。
> 3. **可疑日志**：`compress_conf` 路径每个 chunk 稳定产生 96 条 `quantize_cfg_parser.cpp GenerateIntArrayConfig(95)::"can't find key [shape]"`（无 `compress_conf` 时为 0）。这 96 条 = 24 层 × 4，正对应被激活量化的 24 个 Linear——即 OMG 解析 DOPT 生成的 `quant_params_file` 时未能完整匹配配置。
> 4. **体积旁证**：真机可用的旧 `.om` 为 101.8 MB/chunk，本次 FP16 重编为 101.8 MB/chunk（相差 <0.01%），而 W8A8 版只有 52.1 MB。
>
> 修复：**离线 OM 一律走 FP16**，且按 CLAUDE.md 的要求**直接从原始 HuggingFace 浮点权重导出**（`export-fp16`，不经 DOPT `fake_quant_weight`），OMG 侧 `USE_COMPRESS_CONF=false`。`.mnn`（int8 GPTQ）保持不变，因为它服务于 CPU 与在线 OM 路径，两者本来就正常。
>
> 注意：本仓库 README 里"W8/A16 + 真实校准 cos>0.9997"的结论是在 **host 上用 MNN/eval 脚本**测的，**没有覆盖 OMG `compress_conf` 的真机行为**——host 侧 `eval_chunk_quant.py` 不经过 OMG，因此无法暴露这个问题。这类"host 全绿、真机错"的差异，必须靠真机 `RunSync` 后的激活量级来判断。
>
> **补充排查（2026-10 第二轮，证据更完整）**：为什么"量化了也不该图文无关"——量化误差是**有界**的，不可能把 chunk0 输出从 ~20 放大到 152.75（7.5×），所以只能是**量化参数被错误套用**，而非单纯的精度损失。逐条证据：
>
> 1. **W8A8 从来不是 Kirin9030 的验证配置**。仓库里所有 W8A8 脚本与 README 冒烟测试都写死 `PLATFORM=kirin9020`：
>    `run_real_calib_256_W8A8.sh`、`run_all_chunks_real_calib_W8A8.sh`、README「单 chunk 手动跑」示例。
>    且 `docs/change-summary-20260815.md` §9.3 早已明确标注该矛盾：
>    「`run_all_chunks_real_calib_W8A8.sh` 的 `PLATFORM=kirin9030`：该脚本走 DOPT W8A8 + compress_conf 路径，与 Kirin9030 的 FP16 OMC 策略**矛盾**……否则需要确认。」——这条从未被解决。
> 2. **`act_bit=16` + compress_conf 在 Kirin9030 上根本编不出来**（实测）：`QuantizeOptimizer Fail!` / `anchor_utils.cpp ... The input data anchor is invalid`，无产物。
>    也就是说，README 里"已验证"的 `A16 + unsigned` 口径在 9030 上不可用；能在 9030 编出来的**只有 `act_bit=8`**，而它从未被真机验证过。
> 3. **96 条 `can't find key [shape]` = 24 层 × 4 字段**（chunk0/1/5 一致，无 compress_conf 时为 0）。对应 OMG `quantize_cfg_parser.cpp GenerateIntArrayConfig` 未能完整吃到 DOPT 的 `quant_params_file`。试过 `quant_param_2=True`、给 ONNX 补 `shape_inference`（value_info 0→299）、`--fp16` 改权重 dtype，**96 条均不减少**——说明是 OMG 侧解析不匹配，不是我们漏了哪一项元数据。
> 4. **host 侧模拟（torch 注入 DOPT 记录的真实参数）**：
>    - 正确套用 → chunk0 输出 absmax 20.13、跨图 cos **0.756**（能区分图片，符合预期）
>    - 退化为固定范围（如 [0,1] / [-1,1]）→ 跨图 cos **0.9987 / 0.9996**（**图文无关**，与真机症状同类）
>    - 其他错误模式（漏减 offset、当有符号、直接用整数）→ 放大到 1304 / 4919 / 44235，量级远超真机
>    结论：真机症状属于「参数被错用/退化为固定范围」这一类，而非「参数正确但精度不够」。
> 5. `group_size` 实测**不生效**：`--group_size 128` 与 `--group_size 0` 产出的 `quant_params_file` 字节完全相同（md5 一致）；设 `custom_group_size` 环境变量（官方 `opt_main.py` 的做法）也不改变 `weight_quantizer.s` 粒度（始终 = out_channels，per-channel）。即 `dopt_config` 里写的 `group_size: 128` 没有传到 DOPT。**注意**：per-channel 比 per-group 更细，这条本身不会导致图文无关，不构成根因，但说明「配置写了不等于生效」，不宜再据此推断。
>
> **结论**：Kirin9030 上 `compress_conf` 离线路径属于**未验证组合**，OMG 对 DOPT 量化参数的消费与预期不符，且缺少可用的真机迭代条件。**不要在 9030 上用 W8A8 `compress_conf` 生成视觉 chunk 的离线 OM**；FP16 是当前唯一经真机验证可用的配置。若将来仍要启用 W8A8，需要：在设备侧逐 chunk 比对 OM 与 MNN 的 hidden 输出、确认 OMG 是否真的套用了激活 scale（96 条警告是直接信号），并在拿到可用的官方参考 `compress_conf` 后才能判断是 OMG 版本问题还是参数格式问题。


### 平台差异：Kirin9020 OMC 注意事项

Kirin9020 的 OMC 编译路径与 Kirin9030 有以下关键差异：

1. **不需要 AscendC 环境**：Kirin9020 平台插件（`libai_npucore_tefusion.so` 等）自带算子 tiling 能力，OMG 编译 `--target=omc` 时无需加载 AscendC `set_ascendc_env.sh`。编译时 `LOAD_ASCENDC_ENV` 会保持 `false`。
2. **可用 DOPT W8A8 + compress_conf**：Kirin9020 的 `MatMulV2` 不校验 FP16 权重大小，压缩后的 INT8 buffer 可以正常通过。不需要切换到 FP16 导出路径。
3. **编译输出格式相同**：仍是 `--target=omc` 生成的真正离线图，后缀为 `.omc`。与 Kirin9030 OMC 产物一样，不能用旧在线 IR 的加载方式处理。
4. **产物体积**：W8A8 compress_conf 路径产生的 OMC 文件约 98MB 每 chunk（含压缩权重），而 Kirin9030 FP16 非压缩 OMC 约 49MB 每 chunk。实际大小取决于 chunk 内的 MatMul 数量。

Kirin9020 OMC 编译请直接使用 `run_visual_plugin_matmul_omc.sh` 进行单 route 编译，或参考 `run_all_chunks_real_calib_W8A8.sh` 批量编译所有 chunk。

### 项目换路径后必须重新生成的校准文件

**当前项目绝对路径：`/home/ma-user/workspace/feh/mobiinfer`**（CLAUDE.md 中所有绝对路径均以此为准）。

`transformers/llm/export/plugin_quant_visual_matmul_route_v1/` 下有两个**存档文件**含绝对路径，一旦项目目录改名或搬家就会失效，必须重新生成：

| 文件 | 含绝对路径的字段 | 行数 |
|---|---|---|
| `config_dump_608.json` | `visual_chunk_input_dump_dir` | 1 处 |
| `image_prompt.txt` | 每行 `<img><hw>600,270</hw>{绝对路径}</img>` | 256 行 |

#### 为什么这两处不能写成相对路径

同一个 config 里的键，**解析基准不同**（改之前务必先理解，否则会越改越坏）：

| 配置键 | 代码位置 | 解析基准 |
|---|---|---|
| `llm_model`、`visual_pre_model`、`visual_blocks_chunks` 等 | `transformers/llm/engine/src/llmconfig.hpp:143,147,181` | **config 文件所在目录**（`base_dir_` 前缀） |
| `visual_chunk_input_dump_dir` | `transformers/llm/engine/src/omni.cpp:742-745` | **进程 CWD**（走 `stat`/`mkdir`，无 `base_dir_` 前缀） |
| prompt 里 `<img>...</img>` 的图片路径 | `transformers/llm/engine/src/omni.cpp:1808`（`MNN::CV::imread`） | **进程 CWD** |

`run_real_calib_256.sh:123` 是 `cd "${REPO_ROOT}/build_x86"` 之后才跑 `llm_demo`，所以后两者的相对路径会解析成 `build_x86/xxx`，与脚本里用绝对路径赋值的 `DUMP_RAW_DIR` 对不上，dump 或读图会静默失败。**因此这两处必须写绝对路径**，代价就是换目录后需要重新生成。

#### 重新生成方法

两个文件都在 `transformers/llm/export/plugin_quant_visual_matmul_route_v1/` 目录下执行。

**1) `image_prompt.txt`** —— 图片已在仓库内 `calib_images/`，直接重跑采样脚本：

```bash
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
python3 select_images.py \
  --src_dir ./calib_images \
  --out_prompt ./image_prompt.txt \
  --num 256 --hw 600,270 --seed 42
```

必须保持 `--num 256 --hw 600,270 --seed 42` 不变，否则样本集合或 `seq_len=608` 会变，已量化的 chunk 对不上。

**2) `config_dump_608.json`** —— 以 6chunk 模型目录的 `config.json` 为底，改两个 dump 字段：

> 注意：6chunk 模型目录是**外部依赖，不随仓库备份**，重启/换机后常缺失（旧路径 `/temp/fdh/baiducloud/.../model_6chunk_nor_kirinnpu_visual4` 已不存在）。若目录缺失，请改用下方「路径 B」的 `generate_npz_calib.py`，它只需 HF fp 模型即可产出格式完全一致的校准 npz，无需该目录。

```bash
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
MODEL_CFG_DIR=/path/to/model_6chunk_nor_kirinnpu_visual4   # 外部依赖，按实际位置填写
python3 - "${MODEL_CFG_DIR}/config.json" ./config_dump_608.json "$(pwd)/calib_dump_raw" 256 <<'PY'
import json, sys
src, dst, dump_dir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = json.load(open(src, 'r', encoding='utf-8'))
cfg["visual_blocks_chunk_backends"] = ["cpu"] * len(cfg["visual_blocks_chunks"])
cfg["visual_chunk_input_dump_dir"] = dump_dir
cfg["visual_chunk_input_dump_samples"] = n
json.dump(cfg, open(dst, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
PY
```

改完后校验（两条都必须指向新路径、`/temp` 行数必须为 0）：

```bash
python3 -c "import json;print(json.load(open('config_dump_608.json'))['visual_chunk_input_dump_dir'])"
grep -c '/temp' image_prompt.txt        # 期望 0
grep -cF "$(pwd)/calib_images" image_prompt.txt   # 期望 256
```

#### 注意：这两个文件是存档，实际流程会自行生成到别处

`run_real_calib_256.sh:32,35` 把它们**重新生成到 `/temp`**（`PROMPT_FILE=${CALIB_WORK_ROOT}/image_prompt.txt`、`CONFIG_DUMP=${MODEL_CFG_DIR}/config_dump_608.json`），并不读取仓库里的副本。仓库内这两份的作用是：记录复现参数（`--seed 42 --num 256 --hw 600,270`），以及给不依赖 `llm_demo` 的 `generate_npz_calib.py` 路径做参考。

因此换路径后的真实动作是**按上面命令重新生成**，而不是手改这两个文件。

#### 路径 B：`generate_npz_calib.py` 直接生成校准 NPZ（不依赖 llm_demo）

`generate_npz_calib.py` 走的是**纯 Python 前向**路线：加载 HF fp 模型，过 `patch_embed` 与各 chunk 的 blocks，在 chunk 边界直接存 npz。它**不需要** `llm_demo`、不需要重编引擎、也不需要 2.2G 的 6chunk 目录，只需 HF fp 模型，因此是换机器后最省事的复现路径。

> 该脚本**不使用任何 GPTQ/autoround 量化权重**，只用 fp 权重做前向。量化由下游 `visual_plugin_quant_matmul_route.py` 用 DOPT 完成（见 9030 OMC manifest 的 `"weight_source": "dopt_fake_quant"`）。脚本里曾有的 `--gptq_path` 参数实为死参数（`utils/model.py` 与 `utils/vision.py` 均不消费 `gptq`），已移除。

> ⚠️ **必须复刻 `visual_pre.mnn`，不能只做 `patch_embed`**（2026-10 修复的严重 bug）：
> `visual_pre.mnn` 的 `hidden_states` 输出是 **`patch_embed + pos_embeds`**（见 `llmexport.py::_build_visual_split_wrappers._VisualPre`，
> 以及 HF `Qwen3Vision.forward` 里的 `pos_embeds = pos_embed(idx)*w; hidden += sum(pos_embeds)`），
> 它同时是 chunk0 的输入。脚本若只写 `patch_embed` 就会漏掉 `pos_embeds`。
>
> 实测（同一张图、hw=600,270）：
>
> | 计算 | absmax |
> |---|---|
> | 仅 `patch_embed`（bug 版） | 5.747 |
> | `patch_embed + pos_embeds`（真值） | **32.618** |
>
> 漏项使 chunk0 输入被低估约 5.7 倍 → `min_max` 的 A8 scale 偏小 → 真机激活被 clip，精度下降。
> 且误差会经 chunk0 传播到后续 chunk。**该 bug 不报错**，OMC 照样编译成功，只能靠数值比对发现。
>
> 修复要点（`generate_npz_calib.py`）：`patch_embed` 前先 `view(N, -1)`，再按 `get_idx_weight(grid_thw)` 加 `pos_embeds`。
> 验证方法：新产出的 chunk0 `hidden_states_in` absmax 应≈**32.6**（旧 bug 版≈5.7）；
> 且 chunk1 的 absmax 均值应≈**17.3**，与设备端日志记录的 `hidden absmax=17.4068` 吻合，可作为独立交叉验证。
> 该修复亦说明：**两条路径产出的 npz"格式一致"不代表"数值等价"**，见下。

**`--hw` 尺寸覆盖（与 prompt 的 `<hw>` 标签语义一致）**

引擎侧 `qwen2VisionProcess`（`omni.cpp:982`）的处理是「先把尺寸置为 `<hw>` 的值，再交给 `smartResize` 规整」。脚本的 `--hw` 复刻了同一顺序，故两者结果严格一致：

```
--hw 600,270  ->  smart_resize(600,270, factor=32, 65536, 16777216) = 608x256
              ->  grid 38x16  ->  seq_len = 608
```

生成与 `run_real_calib_256.sh` 格式相同的校准数据集：

```bash
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
python3 generate_npz_calib.py \
  --num_samples 256 \
  --hw 600,270 \
  --output_dir ./calib_npz
```

`--hw` 的语义与参数：

| 参数 | 说明 |
|---|---|
| 不传（默认） | 按图片自然尺寸 `smart_resize`，`seq_len` 随图片变化 |
| `--hw 600,270` | 固定尺寸，产出统一 `seq_len=608`，与 prompt 的 `<hw>600,270</hw>` 等价 |

**必须传 `--hw`，否则多数图片集无法使用**：`calib_images/` 的 495 张图自然尺寸有多种（`704x320`、`608x288`、`672x320`、`384x192`…），自然 `seq_len` 会得到 `880/684/840/288` 等混合值。而导出端 `visual_plugin_quant_matmul_route.py:1085` 是 `export_chunk_onnx(args, samples[0], ...)`——**只用 `samples[0]` 固定 shape 且无 dynamic_axes**，混合 `seq_len` 必然失败。脚本会在**开始前**只读图片头做预检，尺寸不一致时直接报错，不会跑到写出几十 GB 才暴露：

```text
校准样本 seq_len 不一致, 导出端只用 samples[0] 定 shape 会失败:
  seq_len=288 (1 张, 例 ...: 源 345x157 -> 384x192); seq_len=880 (451 张, ...)
```

**参数与产物体积**（供估算）：

| 项 | 值 |
|---|---|
| `--num_samples 16` | 对应 `ENV.md` 的 `CALIB_SAMPLE_COUNT=16`，6 chunk × 16 = 96 个 npz ≈ 144 MB |
| `--num_samples 256` | 6 chunk × 256 = 1536 个 npz ≈ 3.1 GB |
| 每样本 shape | `hidden_states_in (1,608,1024)` / `rotary_pos_emb (2,608,1,64)` / `attention_mask (1,608,608)`，均 fp16 |
| 样本选取 | `sorted(files)[:num_samples]`（**非随机**，与 `select_images.py` 的 `rng.sample(seed=42)` 不同） |

> 注意：两条路径选取的图片子集**不同**（路径 A 的 256 张与路径 B 的 `sorted[:256]` 仅约 126 张重叠）。二者产出的 npz **格式与 shape 完全一致**，可直接用于量化；但若要求与既有 `calib_inputs_256/` 逐样本对齐，需以路径 A 为准，或让脚本改用相同的采样方式。
>
> **"格式一致" ≠ "数值等价"**：路径 A（`llm_demo` dump）dump 的是引擎里 chunk 的实际输入 `preOut[0]`（含 `pos_embeds`）；路径 B 是纯 Python 复刻，必须自己把 `pos_embeds` 补上（见上方 ⚠️）。历史上正是因为只对比了 shape/dtype 就认定两者等价，才漏掉了 `pos_embeds`。
> **切换路径或修改任一路径的前向后，务必做数值交叉验证**（例如比对 chunk1 的 absmax 是否≈17.3，对齐设备日志），不要只比 shape。

**模型权重是外部依赖**：`--model_path`（HF fp 权重）不随仓库备份，可用环境变量覆盖默认值：

```bash
export MOBI_HF_MODEL=/path/to/mobi0402_2B_halfimage_rl
```

#### 相关产物与忽略规则

- `calib_dump_raw/`：`llm_demo` dump 的裸 fp32 bin+meta，约 **6.1 GiB/轮**（每 chunk-样本 ≈4.08 MiB × 6 chunk × 256 样本 ≈ 6144 个文件），可由 `calib_images/` 完全再生，**不入库**。
- `calib_npz/`：`generate_npz_calib.py` 输出的 fp16 npz，同样可再生，**不入库**。

以上两条已在 `plugin_quant_visual_matmul_route_v1/.gitignore` 中忽略。注意 `calib_npz/` 命中既有 `*npz` 规则，会连带忽略其中的 `visual_calib_manifest.json`（如需入库需用 `calib_npz/*` + `!calib_npz/visual_calib_manifest.json` 例外）。

> 磁盘提示：本项目所在盘空间有限，**所有中间产物请输出到 `/temp/work/`**（见 `ENV.md`），不要写进仓库目录。`generate_npz_calib.py` 的 `--output_dir`、OMC 的 `--route_dir`、`llmexport.py` 的 `--dst_path`、以及 x86 构建目录都应指向 `/temp/work/` 下的子目录。

### 端到端已验证流程（Kirin9030 + W8A8，2026-10）

针对 mobi2B：**VIT 走 Kirin9030 离线 OMC 8bit 跑 NPU，LLM 走 GPTQ 8bit 跑 CPU**。完整链路与实测结果如下。

**关键前置**：conda 环境是 `CANN`（大写）；DDK 用 6.1.1.0；`omg` 需 `chmod +x`；AscendC 适配器 wheel 需装入 `tools_ascendc/package/python`（Kirin9030 OMC 必须 `import te_fusion` 成功）。

1. **编译 host 工具**（MNNConvert 供 `llmexport` 使用，llm_demo 供 host 侧验证）：

```bash
mkdir -p /temp/work/build-host-converter && cd /temp/work/build-host-converter
cmake /home/ma-user/workspace/feh/mobiinfer \
  -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON
make -j$(nproc) MNNConvert llm_demo
```

2. **生成校准 npz**（路径 B，只需 HF fp 模型）：见上「路径 B」，16 样本产出 `6×16=96` 个 npz，`seq_len=608`。

3. **逐 chunk 量化 + 导出 ONNX**（W8A8）：

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate CANN
export DDK_DOPT=/temp/huawei-sdk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=$DDK_DOPT:/home/ma-user/workspace/feh/mobiinfer/transformers/llm/export:$PYTHONPATH
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
python visual_plugin_quant_matmul_route.py \
  --route_dir /temp/work/model_omc/chunk0_kirin9030 --chunk_index 0 --npu_chunks 6 \
  --quant_strategy Quant_aigc_ptq --weight_bit 8 --weight_algo min_max \
  --act_bit 8 --input_algo min_max --num_samples 16 --group_size 128 \
  --use_qwen3_style_rotary --input_dir /temp/work/calib_npz --force_regen all
```

4. **Kirin9030 离线 OMC 编译**（必须 `TARGET_MODEL_TYPE=omc` + `LOAD_ASCENDC_ENV=auto`）：

```bash
PLATFORM=kirin9030 TARGET_MODEL_TYPE=omc USE_COMPRESS_CONF=true LOAD_ASCENDC_ENV=auto \
OMG_TOOL=$DDK_PATH/tools/tools_omg/omg OMG_MASTER_DIR=$DDK_PATH/tools/tools_omg/master \
ASCENDC_ENV_SCRIPT=$DDK_PATH/tools/tools_ascendc/set_ascendc_env.sh \
bash run_visual_plugin_matmul_omc.sh /temp/work/model_omc/chunk0_kirin9030 fp16
```

实测 6 个 chunk 全部通过判据：`partition type NPU:1, CPU:0`、`SaveCompiledModelToFile SUCCESS`、`OMG generate offline model success`、`QuantBatchMatmulV3` 出现（≈271 次/chunk），无 `CPU fallback`。产物 **50MB/chunk**（`.omc`）。

5. **导出完整 MNN 模型**（VIT 6 chunk 全 NPU + LLM GPTQ 8bit CPU）：

```bash
cd transformers/llm/export
python llmexport.py \
  --path /temp/models/mobi0402_2B_halfimage_rl \
  --export mnn \
  --gptq_path /temp/models/autoround_export/mobi0402_2B_halfimage_rl-w8g128/ \
  --visual_gptq_path /temp/models/autoround_export/mobi0402_2B_halfimage_rl-w8g128/ \
  --quant_bit 8 --quant_block 128 --visual_quant_bit 8 --visual_quant_block 128 \
  --lm_quant_bit 16 --seperate_embed \
  --visual_split --visual_npu_chunks 6 \
  --visual_chunk_backends "npu,npu,npu,npu,npu,npu" \
  --mnnconvert /temp/work/build-host-converter/MNNConvert \
  --dst_path /temp/work/mnn_export
```

实测 33 文件 / 4.7GB / 0 error；日志含 `Visual GPTQ: replaced 24/24 block weights`（6 chunk 各一次）与 `apply gptq to`（394 次，LLM 侧）。

6. **组装交付目录**：把 `mnn_export/` 的全部文件（含 `llm_config.json`、`*.mnn.json`）复制到
   `/temp/models/mnn_mobi_2B_w8a8_visual_npu_kirin9030/`，并把 6 个 OMC 产物按 app 侧约定
   命名为 `om/visual_blocks_npu_<i>.om`；config 中设 `npu_model_dir: "om"` 与
   `visual_blocks_offline_om: ["om/visual_blocks_npu_<i>.om", ...]`（全部 `npu`）。

**易踩的坑**（本次实际遇到）：

- **`visual_blocks_om_deepstack_dup` 漏配会导致真机闪退**（2026-10 修复）：某个 chunk 的 deepstack 层若正好是该 chunk 的**最后一层**，其 deepstack 输出与 `hidden_states` 是同一个张量。ONNX 导出时用 `Identity` 给 deepstack 命名，而 OMG 会把这个冗余 `Identity` 删掉（编译日志：`onnx_pre_checker.cpp ... "the node Identity dont have output tensor"`），于是该 chunk 的 `.om` **只返回 1 个输出**（其他 deepstack chunk 返回 2 个）。
  - 本模型 24 blocks / 6 chunks = 每 chunk 4 层，`deepstack_visual_indexes=[5,11,17]`；chunk2 覆盖 `[8..11]`，**index 11 恰为其末层** → **只有 chunk2 需要 dup**（chunk1 的 5、chunk4 的 17 都不是各自末层）。
  - 漏配的后果：`allDeepstack` 少一路 → `visual_post` 收到 3 个输入而非期望的 4 个 → `MNN_ASSERT` 在 release 下是**空宏**（`MNNDefine.h:52`），拦不住 → 越界 → **App 闪退，ArkTS `try/catch` 接不住**。
  - 修复：config.json 补 `"visual_blocks_om_deepstack_dup": [2]`。该字段的取值**只取决于 chunk 划分与 deepstack 索引，与校准来源无关**，两套校准（torch/engine）都需要。
  - 判定方法：看各 chunk ONNX 里 deepstack 输出来自 `Add`（独立）还是 `Identity(输入=hidden_states)`（合并）。可脚本化检测：遍历 graph.output，若 deepstack 的名字由 `Identity` 产生且其输入 == hidden_states 输出名，则该 chunk 需要 dup。
  - 安全性质：引擎侧触发条件是 `mOmDeepstackDupIndices.count(i) && omOutputs.size() <= 1`，所以**即使误配也不会重复补**（OM 返回 2 个输出时自动跳过）。
- `llm_config.json` **必须**随模型一起拷贝，缺它会报 `tensor [deepstack_embeds] is input but not found` / `Create module error`，看起来像 deepstack 不匹配，实际只是缺文件。
- 用 symlink 建测试目录后在 symlink 上 `json.dump` 会**穿透改写源文件**；测试 config 要用真实文件。
- `visual_blocks_om_paths` 只存在于文档，**引擎不读该键**；OM 路径由 app 侧 `setNpuChunkExecutor` 注入（见 `omni.hpp:180`），app 按 `visual_blocks_offline_om` / `npu_model_dir` 解析（`mobiinfra-oh/entry/src/main/cpp/napi_init.cpp`）。
- host 上无 hiai NPU，验证时把 `visual_blocks_chunk_backends` 全置 `cpu` 才能跑通视觉前向。**注意：全 CPU 路径走 MNN module，chunk2 的 2 个输出会被保留，因此 host 不会暴露这个 dup 问题——只有真机 OM 路径才会。**

**引擎侧加固**（与本问题配套，`omni.cpp` / `PipelineModule.cpp`）：

- `omni.cpp::qwen2VisionProcess`：调用 `visual_post` 前校验 `blocksOut.size() == postInfo->inputNames.size()`，不匹配则 `MNN_ERROR` 并返回空，避免越界。
- `omni.cpp::qwen2VisionProcess`：`outputs` 为空时提前返回，避免 `outputs[0]` 越界。
- `PipelineModule::onForward`：`mInputSize != inputs.size()` 时打印明确错误并返回空（原先只有 `MNN_ASSERT`，release 下是空操作）；并给 submodule 的输入/输出 stack 索引加范围检查。
- 验证方式：故意构造会少一路 deepstack 的 config，旧二进制越界崩溃，新二进制打印 `visual_post input count mismatch: got 3, expected 4` 并优雅退出（exit 0）。




##环境配置
查看 @ENV.md 文件

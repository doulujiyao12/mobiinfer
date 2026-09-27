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

先阅读仓库根目录的 `ENV.md`，使用其中记录的当前 DDK/CANN/Conda 路径。不要把 `ENV.md` 中的令牌或密码输出到日志、提交或命令行参数。当前 Kirin9030 工具链的关键组成是：

```bash
export DDK_PATH=/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0
source "$DDK_PATH/tools/tools_ascendc/set_ascendc_env.sh"
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

关键组成（详见 `ENV.md`，旧版 6.0.1.0 路径仍保留）：

```bash
export DDK_PATH=/temp/models/csm/DDK-tools-next-6.1.1.0
source "$DDK_PATH/tools/tools_ascendc/set_ascendc_env.sh"   # Kirin9030 必须加载 AscendC
```

首次使用新 DDK 前，需要执行一次 `install.sh`（在 `bash -c` 中 source，zsh 下直接 `source` 会因 `$0` 检测失效）：

```bash
cd "$DDK_PATH/tools/tools_ascendc" && bash -c 'source install.sh'
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

```bash
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
MODEL_CFG_DIR=/temp/fdh/baiducloud/902137265_doulujiyao1/model_6chunk_nor_kirinnpu_visual4
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

**模型权重是外部依赖**：`--model_path`（HF fp 权重）不随仓库备份，可用环境变量覆盖默认值：

```bash
export MOBI_HF_MODEL=/path/to/mobi0402_2B_halfimage_rl
```

#### 相关产物与忽略规则

- `calib_dump_raw/`：`llm_demo` dump 的裸 fp32 bin+meta，约 **6.1 GiB/轮**（每 chunk-样本 ≈4.08 MiB × 6 chunk × 256 样本 ≈ 6144 个文件），可由 `calib_images/` 完全再生，**不入库**。
- `calib_npz/`：`generate_npz_calib.py` 输出的 fp16 npz，同样可再生，**不入库**。

以上两条已在 `plugin_quant_visual_matmul_route_v1/.gitignore` 中忽略。注意 `calib_npz/` 命中既有 `*npz` 规则，会连带忽略其中的 `visual_calib_manifest.json`（如需入库需用 `calib_npz/*` + `!calib_npz/visual_calib_manifest.json` 例外）。


##环境配置
查看 @ENV.md 文件

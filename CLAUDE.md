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
- **Kirin9030（V311）**：编译必须加载 DDK 自带的 AscendC 环境。只设置平台参数但未加载 AscendC 时，MatMul 可能全部被拒绝并回退到 CPU；即使生成了文件，也不能视为成功。该平台稳定方案为离线 OMC + visual chunk 的非压缩 FP16 权重。FP16 Route 必须直接从原始 HuggingFace 浮点权重导出，不能先加载 DOPT `fake_quant_weight.pth` 再转成 FP16；后者保存的是伪量化/反量化后的权重，会改变数值和模型精度。不要对该路径启用旧 W8A8 DOPT `compress_conf`：该配置会令 `MatMulV2` 按 FP16 权重大小校验，而压缩 INT8 buffer 只有预期大小的一半，导致编译失败或错误回退。
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

### 平台差异：Kirin9020 OMC 注意事项

Kirin9020 的 OMC 编译路径与 Kirin9030 有以下关键差异：

1. **不需要 AscendC 环境**：Kirin9020 平台插件（`libai_npucore_tefusion.so` 等）自带算子 tiling 能力，OMG 编译 `--target=omc` 时无需加载 AscendC `set_ascendc_env.sh`。编译时 `LOAD_ASCENDC_ENV` 会保持 `false`。
2. **可用 DOPT W8A8 + compress_conf**：Kirin9020 的 `MatMulV2` 不校验 FP16 权重大小，压缩后的 INT8 buffer 可以正常通过。不需要切换到 FP16 导出路径。
3. **编译输出格式相同**：仍是 `--target=omc` 生成的真正离线图，后缀为 `.omc`。与 Kirin9030 OMC 产物一样，不能用旧在线 IR 的加载方式处理。
4. **产物体积**：W8A8 compress_conf 路径产生的 OMC 文件约 98MB 每 chunk（含压缩权重），而 Kirin9030 FP16 非压缩 OMC 约 49MB 每 chunk。实际大小取决于 chunk 内的 MatMul 数量。

Kirin9020 OMC 编译请直接使用 `run_visual_plugin_matmul_omc.sh` 进行单 route 编译，或参考 `run_all_chunks_real_calib_W8A8.sh` 批量编译所有 chunk。


##环境配置
查看 @ENV.md 文件

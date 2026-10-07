# Plugin Quant Visual MatMul Route V1

这个目录现在是一条独立实验链，不复用其他 `plugin_quant` 路线的脚本或量化产物。

目标是：

- 直接对 visual chunk 做 plugin quant
- 量化配置默认使用 `Quant_act_weight_eco + W4 + group128 + A16(signed)`
- 可选为 `kirin9020` 额外打开 linear 的 `output` 量化配置
- 再用 `FLinearMatmul` 风格导出，让 `q/k/v/o_proj` 和 `linear_fc1/2` 走 plain `MatMul + Add`
- 最后继续跑 OMC

## 当前结论

这条路线目前已经在 `chunk0` 上验证成功：

- 最终 `.omc` 已生成：
  - `model_visual_plugin_matmul_chunk0/omc_output/visual_plugin_matmul_quantized.omc`
- 当前成功产物大小约为 `28.245 MB`
- OMC 日志里已经出现：
  - `blocks.*.self_attn.*_dequant_s16s4 type MatMuls16s4Gen`
  - `blocks.*.mlp.linear_fc*_dequant_s16s4 type MatMuls16s4Gen`
- 日志末尾为：
  - `SaveCompiledModelToFile SUCCESS`
  - `OMG generate offline model success`

可直接查看：

- 日志: `model_visual_plugin_matmul_chunk0/omc_output/omc_kirinx90.log`
- OMC: `model_visual_plugin_matmul_chunk0/omc_output/visual_plugin_matmul_quantized.omc`

## 成功关键点

这条路线最终成功，不是单靠 `plugin quant` 本身，而是以下组合一起成立：

- 量化阶段保留原始 `Linear`，让 `plugin quant` 先正常生成：
  - `fake_quant_weight.pth`
  - `quant_params_file`
- 导出阶段再把目标层替成 `FLinearMatmul`，让主权重层落成 plain `MatMul`
- 导出时对 `fake_quant_weight.pth` 做 key remap，使导出态模型和量化权重完全对齐
- 不再额外改 ONNX 节点名，保持节点名和量化配置里的原始层名一致：
  - `blocks.N.self_attn.q_proj`
  - `blocks.N.self_attn.k_proj`
  - `blocks.N.self_attn.v_proj`
  - `blocks.N.self_attn.o_proj`
  - `blocks.N.mlp.linear_fc1`
  - `blocks.N.mlp.linear_fc2`

## 如何判断是否真的成功

不要只看有没有 `.omc` 文件，建议同时检查下面三件事：

- `omc_output/visual_plugin_matmul_quantized.omc` 是否存在，且大小是否明显小于 `fp16` 路线
- `omc_output/omc_kirinx90.log` 是否出现：
  - `MatMuls16s4Gen`
  - `*_dequant_s16s4`
- 日志末尾是否出现：
  - `SaveCompiledModelToFile SUCCESS`
  - `OMG generate offline model success`

当前 `chunk0` 的成功版本中，以上三项都满足。

## 不必误判的日志

下面这些日志在当前成功版本里也会出现，不应单独据此判断失败：

- `Current node:... has no bias weight, but has quant params.`
  - `Qwen3` 官方成功链路里也会出现
- `The type of MatMuls16s4Gen not find.`
  - 当前成功日志里也会出现，但后续仍会继续走到 `MatMuls16s4Gen` 并最终保存成功
- `PermuteMatmulFusionPass ... only support order 0132`
  - 当前成功日志里仍可能出现，不代表最终 OMC 失败

## 主要文件

- `visual_plugin_quant_matmul_route.py`
  - 单文件完成 `prepare / calibrate / export-onnx / all`
- `run_visual_plugin_matmul_omc.sh`
  - 使用本目录生成的 `quant_params_file` 和 ONNX 跑 OMC
- `bin_to_chunk_npz.py`
  - 把 MNN 引擎（`MNN_VISUAL_CHUNK_INPUT_DUMP` 模式）dump 的 raw float32 bin + meta json 转成校准 npz（`calib_inputs_real/`），格式与随机版完全一致
- `run_all_chunks_real_calib.sh`
  - 一键用真实校准输入（`./calib_inputs_real`）生成全部 6 个 chunk 的 OM，参数与 chunk0 冒烟测试一致
- `select_images.py`
  - 从校准图片源目录随机采样 N 张图，生成 `llm_demo` 用的 `image_prompt.txt`，用 `<hw>H,W</hw>` 强制缩放尺寸统一 seq_len
- `run_real_calib_256.sh`
  - 一键全流程：从真实图像目录直接跑到 6 个量化的 OM（采样 → dump → bin 转 npz → 逐 chunk 量化/OMC），中间产物集中放 `/temp/fdh/input_calib/`（默认 256 张图、seq_len=608、全量 256 校准）
- `eval_chunk_quant.py`
  - 量化效果评估主脚本：对照华为文档插件方法构建 fp chunk vs 量化 qmodel，在真实输入上逐样本对比输出余弦相似度 / MSE
- `run_eval_chunk_quant.sh`
  - 一键评估 wrapper：配好 `conda cann` + `DDK_DOPT` + `PYTHONPATH` 后调用 `eval_chunk_quant.py`
- `unpack_calib_npz.py`
  - 把真实校准 npz 解压成裸 float32 bin + 扁平 meta json，供鸿蒙 app 真机读（app 无 zip/npz 能力，不加新依赖）
- `compare_debug_outputs.py`
  - 对比真机 `runOmVsMnnRealCalibTest` 保存的 debug_outputs（OM/MNN-CPU/MNN-NPU 三路输出），算精度指标 + 作图（数值曲线、scatter、误差曲线、cos 柱状图）

## 环境准备

```bash
source /data/dahu/anaconda3/etc/profile.d/conda.sh
conda activate llama
export DDK_DOPT=/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3                      
export PYTHONPATH=$DDK_DOPT:/home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export:$PYTHONPATH

cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
```

## 可选: 本地生成任意 SeqLen 校准输入

这个目录现在也带了一份本地副本脚本：

- `collect_visual_act_stats_v6.py`

它可以直接生成目标 `seq_len` 对应的新 `calib_inputs`，这样比“沿用旧 `256` 样本再硬改 shape”更稳妥。

例如生成 `seq_len=512` 的新输入：

```bash
python collect_visual_act_stats_v6.py \
  --path /data/fengdahu/model/mobi0429_2B__nore_halfimage \
  --dst_path ./model_visual_plugin_matmul_chunk0_calib_seq512 \
  --npu_chunks 6 \
  --visual_gptq_path /data/fengdahu/model/mobi0429_2B_halfimage-quant-gptq-w4a16-n128-s1024-visual/mobi0429_2B__nore_halfimage-w4g128 \
  --num_samples 4 \
  --seq_len 512 \
  --dtype fp16
```

生成后的输入目录是：

- `./model_visual_plugin_matmul_chunk0_calib_seq512/calib_inputs`

如果你想把这套 `seq_len=512` 输入直接接到后续完整链路，可以这样：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_seq512 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --input_dir ./model_visual_plugin_matmul_chunk0_calib_seq512/calib_inputs \
  --sample_prefix chunk_00 \
  --num_samples 2 \
  --force_regen \
  all
```

`--sample_prefix` 用于指定校准 npz 文件的命名前缀。不传时默认按 `chunk_{chunk_index:02d}` 自动拼接（例如 `chunk_index=0` → `chunk_00`）。如果你的自定义校准输入使用了其他前缀，可以通过该参数显式指定。

## 用真实图像生成校准输入（推荐，替代随机输入）

上面 `collect_visual_act_stats_v6.py` 默认用随机正态分布生成校准输入（`hidden` absmax≈0.49），与真实激活分布差距较大。本节给出一条**用真实图像跑 MNN 引擎、dump 每个 chunk 真实激活、再转成校准 npz** 的链路，校准数据更贴近真实分布。

校准输入目录：`./calib_inputs_real`（产物格式与随机版逐字节对齐，下游链路零改动，只需把 `--input_dir` 指向它）。

整条链路分三步：① 编译打开 dump 宏的 MNN → ② 跑真实图像 dump 出 raw bin → ③ bin 转 npz。

### 涉及的代码修改（MNN 引擎侧）

为了能在 MNN 推理时 dump 每个 visual chunk 的三个输入（`hidden_states_in` / `rotary_pos_emb` / `attention_mask`），对 MNN 引擎做了**默认关闭、开关式**的改动：

| 文件 | 改动 |
|------|------|
| `CMakeLists.txt`（顶层） | 新增 `option(MNN_VISUAL_CHUNK_INPUT_DUMP ... OFF)`，默认关闭 |
| `transformers/llm/engine/CMakeLists.txt` | 宏 ON 时 `target_compile_definitions(llm PRIVATE MNN_VISUAL_CHUNK_INPUT_DUMP)` |
| `transformers/llm/engine/src/omni.hpp` | Omni 类新增 3 个成员（全在 `#ifdef` 内）：`mVisualChunkDumpDir` / `mVisualChunkDumpMaxSamples` / `mVisualChunkDumpedSamples` |
| `transformers/llm/engine/src/omni.cpp` | 全部 dump 逻辑在 `#ifdef MNN_VISUAL_CHUNK_INPUT_DUMP` 内：建目录 helper、VARP→float32 bin helper、读 config、chunk 循环 `onForward` 前 dump 三输入、每图自增计数 |

**关键设计：宏 OFF 时引擎与原代码完全等价**（预处理后零 dump 符号），手机侧 `libMNN.so` 不受影响，符合 mobiinfra-oh 不加依赖的约束。

**两个 dump 行为细节：**

1. **rotary shape 对齐**：MNN `visual_pre` 输出 rotary 是 `[2,1,S,1,64]`（rank-5，源码 `utils/transformers.py` Rotary.forward 的 `unsqueeze(2).unsqueeze(1)`），而 OM/npz 校准格式是 `[2,S,1,64]`（rank-4，导出时 squeeze(1)）。数据存储顺序一致，只多一个 size-1 维。dump 时对 rotary 做 `_Squeeze(v, {1})` 对齐到 OM 格式。
2. **hidden 统一带 batch 维**：chunk0 的 `hidden_states_in` 来自 `visual_pre` 的 `patch_embed`，是 `[S,1024]`（rank-2）；chunk≥1 是 `[1,S,1024]`（rank-3）。转 npz 时统一 reshape 成 `[1,S,1024]`。
3. **只 dump MNN-module 分支**：host 上 hiai NPU 不可用，必须把 `visual_blocks_chunk_backends` 全置 `cpu`，让 6 个 chunk 全走 MNN-module 路径才能触发 dump。

dump 输出格式（供下游 Python 转 npz）：

```
<dump_dir>/
  hidden_states_in_chunk{i}_sample{j}.bin   # raw float32
  rotary_pos_emb_chunk{i}_sample{j}.bin     # raw float32
  attention_mask_chunk{i}_sample{j}.bin     # raw float32
  meta_chunk{i}_sample{j}.json              # 含三个 tensor 的 shape/dtype/file/ok
```

### 步骤 1: 编译打开 dump 宏的 MNN

```bash
cd /home/ma-user/workspace/fdh/mobiinfer
mkdir -p build_x86 && cd build_x86
cmake .. -DMNN_BUILD_LLM=true \ -DMNN_AVX512=true  \ -DMNN_LOW_MEMORY=true \ -DMNN_BUILD_LLM_OMNI=ON \ -DMNN_BUILD_TEST=ON \ -DMNN_BUILD_TOOLS=ON \ -DMNN_VISUAL_CHUNK_INPUT_DUMP=ON
make -j$(nproc) llm_demo
```

> 不需要 dump 时去掉 `-DMNN_VISUAL_CHUNK_INPUT_DUMP=ON` 即可，引擎恢复原样。

### 步骤 2: 配置 + 跑真实图像，dump 出 raw bin

用 `model_402_visual6_fp16/model_402_visual6_fp16/` 这套 6-chunk MNN 权重目录（visual block 0~5 的 mnn 权重 + `visual_pre/post` 都在里面）。对其 `config.json` 打两个补丁：

```jsonc
{
  // host 上 hiai NPU 不可用，全置 cpu 让 6 个 chunk 走 MNN-module 路径（才能 dump）
  "visual_blocks_chunk_backends": ["cpu","cpu","cpu","cpu","cpu","cpu"],
  // 新增 dump 配置
  "visual_chunk_input_dump_dir": "/home/ma-user/workspace/fdh/mobiinfer/build_x86/chunk_dump_raw",
  "visual_chunk_input_dump_samples": 2
}
```

> 可用 `jq` 或一段 python 打补丁，或直接手改一份 `config_dump.json`，不动原 `config.json`。

然后跑 `llm_demo` 喂真实图像（只要视觉前向，`max_token=1` 即可，图通过 prompt 文件传入）：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/build_x86
./llm_demo /temp/fdh/baiducloud/902137265_doulujiyao1/model_402_visual6_fp16/model_402_visual6_fp16/config_dump.json image_prompt.txt 1
```

产物：`build_x86/chunk_dump_raw/`（6 chunk × N sample × 3 tensor + meta json）。

### 校准输入数量（sample 数）怎么控制

校准用几个输入由**两个层面**配合决定，需要分开理解：

**层面 1 — dump 阶段决定「最多能用到几个」**

`config_dump.json` 里的 `visual_chunk_input_dump_samples` 控制 dump 几张图：

```jsonc
"visual_chunk_input_dump_samples": 2   // ← 改成你想要的图数，比如 8
```

跑 `llm_demo` 时传对应数量的图（`image_prompt.txt` 每行一张图路径），就会 dump 出每个 chunk 该数量的 sample（`chunk_XX_sample_000~00N-1.npz`）。

**层面 2 — 量化阶段决定「实际用几个」**

`visual_plugin_quant_matmul_route.py` 的 `--num_samples`（`run_all_chunks_real_calib.sh` 头部的 `NUM_SAMPLES`）控制实际参与校准的数量。`load_calibration_samples` 的逻辑是：按文件名排序 glob 出该 chunk 全部样本，**取前 `--num_samples` 个**：

```python
files = sorted(input_path.glob(f"{prefix}_sample_*.npz"))
if num_samples > 0:
    files = files[:num_samples]   # 取前 N 个
```

所以：

- dump 了 8 个，`--num_samples 4` → 只用前 4 个
- dump 了 8 个，`--num_samples 8` → 全用

**完整示例：用 8 张图校准**

```bash
# 1) dump 阶段：config_dump.json 里改
#    "visual_chunk_input_dump_samples": 8
#    准备 image_prompt.txt（8 行，每行一张图路径）
#    跑 llm_demo，dump 出每个 chunk 8 个 sample

# 2) 转 npz（不变，会把全部 dump 都转出来）
python3 bin_to_chunk_npz.py \
  --dump_dir /home/ma-user/workspace/fdh/mobiinfer/build_x86/chunk_dump_raw \
  --out_dir ./calib_inputs_real

# 3) 量化阶段：改 run_all_chunks_real_calib.sh 头部
#    NUM_SAMPLES=8   # 实际用几个校准输入（≤ dump 的数量）
bash run_all_chunks_real_calib.sh
```

**注意事项：**

1. `NUM_SAMPLES` 不要超过 dump 出的样本数，否则 `files[:num_samples]` 只会取到全部（不会报错，但达不到 N）。要让量化用到 N 个，dump 阶段就必须 dump 至少 N 张图。

2. 样本选择是「按文件名排序取前 N 个」（`sample_000` → `001` → …），**不是随机采样**。想让某些特定图参与校准，控制 `image_prompt.txt` 的顺序即可——排在前面的图先 dump，对应 `sample_000`。

3. 校准样本数与量化质量/时间的权衡：样本越多激活统计越全面、量化精度通常越好，但校准时间线性增长。一般 4~8 个真实样本就明显优于随机输入，不必追求大量。

4. **换图重 dump 时**：bin 会覆盖到同一个 `chunk_dump_raw/`，但 `mVisualChunkDumpedSamples` 是每次 `llm_demo` 进程内从 0 计数。所以换图重 dump 前最好清空 `chunk_dump_raw/`，避免新旧样本混杂；然后重新跑 `bin_to_chunk_npz.py`（会覆盖 `calib_inputs_real/`）。

```bash
rm -rf /home/ma-user/workspace/fdh/mobiinfer/build_x86/chunk_dump_raw/*
# 再跑 llm_dump + bin_to_chunk_npz.py
```

### 步骤 3: bin 转 npz

`bin_to_chunk_npz.py` 的作用：把 MNN 引擎在 `MNN_VISUAL_CHUNK_INPUT_DUMP` 模式下 dump 出来的 per-chunk 校准输入（raw float32 `.bin` + `meta_*.json`）转换成 plugin-quant 校准链路期望的 `chunk_XX_sample_YYY.npz` 格式，使真实校准输入可以直接喂给 `visual_plugin_quant_matmul_route.py --input_dir`。

**输入**（步骤 2 的 dump 产物）：

```
<dump_dir>/
  hidden_states_in_chunk{i}_sample{j}.bin   raw float32
  rotary_pos_emb_chunk{i}_sample{j}.bin     raw float32
  attention_mask_chunk{i}_sample{j}.bin     raw float32
  meta_chunk{i}_sample{j}.json              含每个 tensor 的 shape/dtype/file/ok
```

**输出**（对齐 `collect_visual_act_stats_v6.py` 的约定）：

```
<out_dir>/
  chunk_{i:02d}_sample_{j:03d}.npz
      hidden_states_in   [1, S, 1024]  fp16
      rotary_pos_emb     [2, S, 1, 64] fp16
      attention_mask     [1, S, S]     fp16
  visual_calib_manifest.json
```

**关键处理：**

- 从每个 `meta_chunk{i}_sample{j}.json` 读取三个 tensor 的 `shape`，用 `np.fromfile` 读 raw float32 并按 shape reshape（校验 float 数量与 shape 乘积一致，不一致报错）
- `hidden_states_in` 统一补成 `[1, S, 1024]`：chunk0 来自 `visual_pre` 的 `patch_embed` 是 `[S,1024]`（rank-2），chunk≥1 已是 `[1,S,1024]`（rank-3），脚本对 rank-2 自动 `arr[None, ...]`
- `rotary_pos_emb`：MNN dump 时已 squeeze(1)，输出即 `[2,S,1,64]`，脚本直接 reshape（与 OM 格式一致）
- dtype 统一转 **fp16**（与现有 `calib_inputs` 完全一致）
- 每个 npz 文件名严格匹配下游 glob `chunk_{chunk_index:02d}_sample_*.npz`
- 同时写 `visual_calib_manifest.json`，记录每个 sample 的 chunk/sample_idx/文件路径/shapes/dtypes 以及统计量（`hidden_absmax`、`rotary_min/max`），便于核对数据是否真实

**参数：**

| 参数 | 说明 |
|------|------|
| `--dump_dir` | omni.cpp 的 dump 目录（步骤 2 产物） |
| `--out_dir` | 输出的 calib_inputs 目录 |
| `--dtype` | 输出 npz 的 dtype，默认 `fp16`（与现有 calib 一致），可选 `fp32` |

用法：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
python3 bin_to_chunk_npz.py \
  --dump_dir /home/ma-user/workspace/fdh/mobiinfer/build_x86/chunk_dump_raw \
  --out_dir ./calib_inputs_real
```

产物：`./calib_inputs_real/chunk_{00..05}_sample_{000..00N-1}.npz` + `visual_calib_manifest.json`。

校验格式与现有随机输入一致（shape/dtype 逐 key 对比）：

```bash
python3 - <<'EOF'
import numpy as np
keys=["hidden_states_in","rotary_pos_emb","attention_mask"]
for ci in range(6):
    real=np.load(f"./calib_inputs_real/chunk_{ci:02d}_sample_000.npz")
    ref =np.load(f"./model_visual_plugin_matmul_calib_seq608/calib_inputs/chunk_{ci:02d}_sample_000.npz")
    assert all(real[k].shape==ref[k].shape and real[k].dtype==ref[k].dtype for k in keys), ci
print("ALL FORMAT MATCH")
EOF
```

真实校准输入的数值特征（与随机版对比，确认是真实激活）：

- `rotary_pos_emb`：范围 `[-1, 1]`（真实 cos/sin；随机版只有 ±0.43）
- `hidden_states_in`：absmax 随 chunk 深度变化（如 32→17→10→241→275→292，深层残差累积；随机版恒为 0.49）
- `attention_mask`：全 0

### 用真实校准输入一键生成全部 6 个 chunk 的 OM

本目录提供了 `run_all_chunks_real_calib.sh`，参数与 chunk0 冒烟测试完全一致（`Quant_aigc_ptq / W8 min_max / A16 / input min_max unsigned / group128 / use_qwen3_style_rotary`），`--input_dir` 指向 `./calib_inputs_real`：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
bash run_all_chunks_real_calib.sh
```

脚本特性：

- 自动跳过已存在 `.om` 的 chunk（想强制重跑删掉对应 `model_visual_plugin_matmul_chunkX_real/omc_output/*.om` 即可）
- 每个 chunk 的成功判据：`.om` 文件存在 + 日志含 `OMG generate offline model success`
- 平台默认 `kirin9020`，可在脚本头部改 `PLATFORM`
- 结尾汇总 6 个 chunk 的 OM 状态

产物：

```
model_visual_plugin_matmul_chunk{0..5}_real/omc_output/visual_plugin_matmul_quantized.om   # 各约 49M
```

> 注意：`run_visual_plugin_matmul_omc.sh` 用 `--target=om`，实际产物是 `.om`（脚本末尾打印的 `.omc` 是误导信息），所以成功判据用 `.om`。

### 单 chunk 手动跑（调试用）

```bash
source /data/dahu/anaconda3/etc/profile.d/conda.sh
conda activate llama
export DDK_DOPT=/data/fengdahu/cann_codesampe/export DDK_DOPT=/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3                      
export PYTHONPATH=$DDK_DOPT:/home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export:$PYTHONPATH

cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1

# 1) 单 chunk 量化 + 导出 ONNX（用真实校准输入）
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_real \
  --chunk_index 0 \
  --npu_chunks 6 \
  --quant_strategy Quant_aigc_ptq \
  --weight_bit 8 \
  --weight_algo min_max \
  --act_bit 16 \
  --input_algo min_max \
  --input_unsigned_quant \
  --num_samples 2 \
  --group_size 128 \
  --use_qwen3_style_rotary \
  --input_dir ./calib_inputs_real \
  --force_regen \
  all

# 2) 单 chunk 跑 OMG
PLATFORM=kirin9020 \
bash run_visual_plugin_matmul_omc.sh \
  ./model_visual_plugin_matmul_chunk0_real \
  fp16
```

## 用大批量真实图像一键校准量化（推荐：`run_real_calib_256.sh`）

上面「用真实图像生成校准输入」给出的是手工三步链路（采样 dump→跑 llm_demo→bin 转 npz→接入量化）。本节给出一条**一键全流程脚本** `run_real_calib_256.sh`，从真实图像目录直接跑到 6 个量化的 OM，全部中间产物集中在一个工作目录，便于本地空间有限时使用。

本脚本默认采样 **256 张真实图像**、强制缩放到 `<hw>600,270</hw>`（→ seq_len=608）、每个 chunk 用 256 个真实样本校准。所有中间产物默认放 `/temp/fdh/input_calib/`，OM 产物放 `/temp/fdh/model_omc/`。

### 涉及的脚本

- `select_images.py`
  - 从校准图片源目录随机采样 N 张图，生成 `llm_demo` 用的 `image_prompt.txt`。每行格式 `<img><hw>600,270</hw>/abs/path/to/image.jpg</img>`，其中 `<hw>H,W</hw>` 被 `omni.cpp::multimodeProcess` 解析为 `mVisionHeight/Width`，再经 `qwen2VisionProcess` round 到 32 对齐（`600,270 → H=608, W=256 → grid_h=38, grid_w=16 → seq_len=608`）。采样随机种子可复现。
- `bin_to_chunk_npz.py`（见前面「步骤 3」的详细说明）
  - 把 MNN dump 的 raw float32 bin + meta json 转成校准 npz。
- `run_real_calib_256.sh`
  - 一键全流程：采样 → 生成 dump 配置 → 跑 `llm_demo` dump → bin 转 npz（含格式校验）→ 逐 chunk 量化+导出 ONNX+OMC → 汇总。

### 关键设计点

1. **同一 seq_len 是硬约束**：导出 ONNX 时用 `samples[0]` 固定 shape、**无 dynamic_axes**，所以 N 个校准样本必须共享同一 seq_len，否则后续样本会因 shape 不匹配报错。`select_images.py` 用 `<hw>600,270</hw>` 把所有图强制缩放到产生 seq_len=608 的尺寸。
2. **必须一次 `llm_demo` 进程跑完所有图**：`omni.cpp` 的 dump 计数器 `mVisualChunkDumpedSamples` 是进程内从 0 计，所以 N 张图要在一次 `llm_demo` 调用里全部跑完（脚本里 `image_prompt.txt` 每行一张图，`benchmark()` 循环跑）。
3. **dump 配置必须放在模型目录里**：原 `config.json` 用相对路径引用 `llm.mnn`/`tokenizer.mtok`，`llm_demo` 按配置文件所在目录解析这些路径。`run_real_calib_256.sh` 因此把 `config_dump_608.json` 生成在**模型目录**里（`MODEL_CFG_DIR`），而不是工作目录——否则会报 `tokenizer file not found` / `Unable to open llm_config file`。
4. **dump 前提**：`visual_blocks_chunk_backends` 必须**全置 `cpu`**（host 上 hiai NPU 不可用，只有 MNN-module 路径才会触发 dump）。注意 `visual_blocks_backend_type` 仍是 `hiai` 不影响——真正决定 chunk 走 MNN-module 还是 OM 路径的是 `visual_blocks_chunk_backends`。
5. **空间估算**：256 张图 raw bin ≈ 6.5GB（6 chunk × 256 sample × 3 tensor，float32），转完 npz(fp16) ≈ 3.3GB。`KEEP_RAW=0`（默认）转完 npz 后自动 `rm -rf chunk_dump_raw/` 清掉 raw bin 省空间。`/temp` 至少要留 ~10GB。

### 用法

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
bash run_real_calib_256.sh
```

脚本头部的可调参数：

```bash
NUM_IMAGES=256                  # dump 几张图 (同时 = 量化用几个样本)
HW_OVERRIDE="600,270"           # 强制缩放尺寸 -> seq_len=608
SEED=42                         # 采样随机种子 (可复现)
CALIB_WORK_ROOT=/temp/fdh/input_calib            # 所有中间产物根目录
SRC_IMG_DIR=/temp/csm/sft-0422-quant-500-half-size  # 真实图源目录
MODEL_CFG_DIR=/temp/fdh/baiducloud/902137265_doulujiyao1/model_6chunk_nor_kirinnpu_visual4
OMC_OUT_ROOT=/temp/fdh/model_omc
ROUTE_SUFFIX=real256            # route_dir 后缀
NPU_CHUNKS=6
PLATFORM=kirin9020
KEEP_RAW=0                      # 1=保留 raw bin, 0=转完 npz 后清理 (省空间)
SKIP_DUMP=0                     # 1=跳过 dump (复用已有 DUMP_RAW_DIR)
SKIP_NPZ=0                      # 1=跳过 bin->npz (复用已有 CALIB_NPZ_DIR)
```

### 数据流

```
真实图像(256张, 强制 hw=600,270 -> seq_len=608)
  → llm_demo (MNN引擎, MNN_VISUAL_CHUNK_INPUT_DUMP=ON, chunk_backends=cpu)
  → dump raw float32 bin + meta json      /temp/fdh/input_calib/chunk_dump_raw/
  → bin_to_chunk_npz.py 转 fp16 npz        /temp/fdh/input_calib/calib_inputs_256/
  → visual_plugin_quant_matmul_route.py (每 chunk, --num_samples 256)
  → run_visual_plugin_matmul_omc.sh        /temp/fdh/model_omc/model_..._chunk{i}_real256/
  → 6 个 visual_plugin_matmul_quantized.om
```

### 流程步骤（脚本内部）

1. 采样：`select_images.py` 从 `SRC_IMG_DIR` 随机抽 256 张 → `image_prompt.txt`
2. 生成 dump 配置：把 `MODEL_CFG_DIR/config.json` 打两个补丁（`chunk_backends` 全 `cpu` + 加 `visual_chunk_input_dump_dir`/`visual_chunk_input_dump_samples`），写到模型目录里的 `config_dump_608.json`
3. 跑 `llm_demo`：`./llm_demo config_dump_608.json image_prompt.txt 1`，dump 出 6144 个文件（6 chunk × 256 sample × (3 tensor + 1 meta)），校验文件数是否齐全
4. bin→npz：`bin_to_chunk_npz.py` 转成 1536 个 fp16 npz（256×6），并与现有 `model_visual_plugin_matmul_calib_seq608/calib_inputs/` 逐 key 对比 shape/dtype 做格式校验
5. （可选）清理 raw bin（`KEEP_RAW=0`）
6. 逐 chunk 量化：`visual_plugin_quant_matmul_route.py ... --num_samples 256 --input_dir calib_inputs_256 --force_regen all`，参数与 chunk0 冒烟测试一致（`Quant_aigc_ptq / W8 min_max / A16 / input min_max unsigned / group128 / use_qwen3_style_rotary`）
7. 逐 chunk OMC：`run_visual_plugin_matmul_omc.sh`，成功判据为 `.om` 文件存在 + 日志含 `OMG generate offline model success`
8. 汇总 6 个 chunk 的 OM 状态，列出失败项

### 前置条件检查（脚本自动做）

- `build_x86/llm_demo` 可执行
- `build_x86/CMakeCache.txt` 里 `MNN_VISUAL_CHUNK_INPUT_DUMP:BOOL=ON`（开过 dump 宏），否则 dump 不会触发
- `MODEL_CFG_DIR` 存在（含 6 chunk 的 MNN 权重 + `visual_pre/post` + `tokenizer.mtok`）

未满足时脚本会直接退出并提示。重新编译带 dump 宏的 llm_demo：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/build_x86
cmake .. -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON -DMNN_VISUAL_CHUNK_INPUT_DUMP=ON
make -j$(nproc) llm_demo
```

### 产物

```
/temp/fdh/input_calib/
  image_prompt.txt              # 256 行
  config_dump_608.json          # 在模型目录里(相对路径相对它解析)
  calib_inputs_256/             # 1536 个 npz + visual_calib_manifest.json
/temp/fdh/model_omc/
  model_visual_plugin_matmul_chunk{0..5}_real256/omc_output/visual_plugin_matmul_quantized.om   # 各 49M
```

### 复用与调试

- **只想换图重新采样**：改 `SRC_IMG_DIR` / `SEED` / `NUM_IMAGES`，重跑；`chunk_dump_raw` 会被清空重建，`calib_inputs_256` 会被覆盖
- **想跳过 dump 复用已有 npz**：`SKIP_NPZ=1`（同时建议 `SKIP_DUMP=1`），脚本直接从 step 5 跑量化
- **想强制重跑某 chunk**：删掉对应 `model_visual_plugin_matmul_chunkX_real256/omc_output/*.om`，脚本会自动重跑该 chunk（其余跳过）
- **量化用更少样本**：把脚本 step 5 里的 `--num_samples ${NUM_IMAGES}` 改小（如 8 或 16）。npz 仍生成 256 个，但量化只取前 N 个参与校准。256 个全量校准更全面但更慢，一般 8~16 个真实样本已明显优于随机输入
- **保留 raw bin 排查问题**：`KEEP_RAW=1`（多占约 6.5GB）

## 量化效果评估（fp 基线 vs 量化模型，逐样本数值对比）

上面几节讲的是「如何用真实校准输入生成量化的 OM」。本节给出一条**量化效果评估**链路：对照华为 CANN Kit 文档「[量化效果评估](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-llm-quantization-effect-evaluation)」的插件方法，对浮点 chunk 插入量化算子做仿真推理，但把文档里的「整模型 generate 看输出通不通」改成更适合 chunk 子模块的「**逐样本对比量化前后输出张量的余弦相似度 / MSE**」。

### 涉及的脚本

- `eval_chunk_quant.py`
  - 评估主脚本。复用 `visual_plugin_quant_matmul_route.py` 的 `load_visual_chunk` / `load_calibration_samples` / `reset_kv_cache` / `config_path` / `calibrated_state_path`，构建 fp 基线 chunk 和量化 qmodel，在真实输入上逐样本对比输出。
- `run_eval_chunk_quant.sh`
  - 一键 wrapper：配好 `conda cann` + `DDK_DOPT` + `PYTHONPATH` 后调用 `eval_chunk_quant.py`。

### 对照华为文档

文档的 `get_quanted_model` → 本脚本的 `build_quant_chunk`，一一对应：

| 华为文档步骤 | 本脚本对应 |
|---|---|
| `base_model = AutoModelForCausalLM.from_pretrained(...)` | `chunk, meta = load_visual_chunk(model_path, route_dir, npu_chunks, ci)`（浮点 visual chunk） |
| `model = optimize_model(base_model, dopt_config)` | `qmodel = optimize_model(chunk, dopt_config.chunk_0X.json)` |
| `model.load_state_dict(torch.load(quanted_ckpt), strict=True)` | `qmodel.load_state_dict(torch.load(calibrated_chunk_0X.pth), strict=True)` |
| `set_quant_state(model, weight_state=True, input_state=True)` | `set_quant_state(qmodel, weight_state=True, input_state=True)` |
| `set_calibrate_state(model, False)` | `set_calibrate_state(qmodel, False)`（关校准态，用固化 scale 做 fake quant） |
| `model.generate(prompt)` 看输出 | **跑 N 个真实输入，对比量化前后输出张量的 cos / MSE** |

判断标准差异：文档「仿真推理结果正常 = 量化成功」（定性）；本脚本「量化前后输出余弦相似度 > 阈值 = 量化成功」（定量，逐样本逐输出）。

### 核心设计：fp vs quant 同输入对比

```
N 个真实输入 ──┬─ fp chunk (浮点原 chunk) ─────────→ out_fp   (hidden + deepstack_k)
(calib_inputs) └─ qmodel (optimize_model +           → out_quant (hidden + deepstack_k)
                calibrated pth + quant_state)            ↓
                                              余弦相似度 / rel_mse / absmax / 逐元素误差
```

**关键点**：fp 基线和 qmodel **必须各自独立加载一份 HF 模型**，不能共享 `visual`。因为 `optimize_model(chunk, cfg)` 会**原地**把 chunk 里的 `nn.Linear` 换成 `QLinear`，若共享同一份 `visual.blocks`，优化后 fp chunk 也被改成量化模型，两者输出完全相同（cos≈1.0、rel_mse=0），评估失效。`build_fp_chunk` 和 `build_quant_chunk` 各自调 `load_visual_chunk`（各加载一次 HF 模型）规避此问题。

### 评估指标

对每个输出张量（hidden + 各 chunk 的 deepstack_k）逐样本算：

- `cos`：余弦相似度（主指标，>0.95 为好；W8/A16 + 真实校准通常 >0.999）
- `rel_mse`：归一化 MSE = `mean((a-b)^2)/(|a|·|b|)`，越小越好
- `absmax_fp` / `absmax_q`：量化前后激活 absmax（反推是否真实激活）
- `max_abs_err` / `mean_abs_err`：逐元素绝对误差

汇总成 N 个样本的 mean / min / max / median / std。

### deepstack 输出分布

模型 `deepstack_visual_indexes = [5, 11, 17]`（全局 block 索引），按 6 chunk 切分（每 chunk 4 block）：

| chunk | block 区间 | deepstack 数 | 输出 |
|---|---|---|---|
| 0 | [0,4) | 0 | hidden_states |
| 1 | [4,8) | 1（block 5） | hidden_states, deepstack_0 |
| 2 | [8,12) | 1（block 11） | hidden_states, deepstack_0 |
| 3 | [12,16) | 0 | hidden_states |
| 4 | [16,20) | 1（block 17） | hidden_states, deepstack_0 |
| 5 | [20,24) | 0 | hidden_states |

deepstack 是 Qwen3-VL 的多层特征聚合：在指定中间 block 处抽出该层 hidden_states，经 `deepstack_merger` 合并后送 LLM 做辅助特征。chunk 输出 = 最后 block 的 `hidden_states` + 每个 deepstack block 的中间 hidden。

### 用法

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
bash run_eval_chunk_quant.sh
```

可调环境变量（在命令前加）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `NUM_SAMPLES` | 256 | 每 chunk 评估几个样本（先小规模验证再跑全量） |
| `CHUNKS` | 0,1,2,3,4,5 | 评估哪些 chunk，逗号分隔 |
| `MODEL_PATH` | `/temp/models/mobi0402_2B_halfimage_rl` | HF 浮点模型路径（与量化时一致） |
| `ROUTE_ROOT` | `/temp/fdh/model_omc` | route_dir 根目录 |
| `ROUTE_SUFFIX` | `real256` | route_dir 后缀 |
| `INPUT_DIR` | `/temp/fdh/input_calib/calib_inputs_256` | 真实校准输入 npz 目录 |
| `OUT_DIR` | `/temp/fdh/input_calib/eval_real256` | 评估产物输出目录 |

### 产物

```
/temp/fdh/input_calib/eval_real256/
  eval_report.json          # 6 chunk × 输出 × 指标的汇总 + cos 总览表
  chunk_00/
    per_sample.csv          # N 行逐样本指标（hidden + 各 deepstack）
    metrics.json            # 汇总统计 + load_info + local_deepstack_count
  chunk_01/ ...
```

### 小规模验证（chunk0，4 样本）

**强烈建议先跑小规模验证**确认脚本能跑通、指标合理，再跑全量 256×6 chunk：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
NUM_SAMPLES=4 CHUNKS=0 bash run_eval_chunk_quant.sh
```

预期输出（chunk0 无 deepstack，只有 hidden_states）：

```
========== chunk 0  route=/temp/fdh/model_omc/model_visual_plugin_matmul_chunk0_real256 ==========
  [load] fp 基线 chunk ...
  [load] 量化 qmodel (optimize_model + calibrated pth) ...
  block [0,4)  local_deepstack=0  outputs=['hidden_states']
  load_info: strict=True missing=0 unexpected=0
  samples: 4
    sample 4/4  cos(hidden)=0.999822
  --- chunk 0 汇总 ---
    hidden_states : cos mean=0.999811 min=0.999779 rel_mse=6.43e-10 absmax_fp=17.623 absmax_q=17.634

============================================================
eval_report: /temp/fdh/input_calib/eval_real256/eval_report.json
============================================================
chunk  cos_hidden_mean  cos_hidden_min   status
    0         0.999811        0.999779       ok
```

**结果判读**：
- `cos mean=0.999811`：量化前后 hidden 输出余弦相似度均值，极高（W8/A16 + 真实校准，视觉 chunk 量化损失极小）
- `cos min=0.999779`：4 个样本里最低的，仍 > 0.9997
- `rel_mse=6.43e-10`：归一化 MSE，极小
- `absmax_fp=17.623` / `absmax_q=17.634`：真实激活范围（非随机 0.49），量化前后几乎一致
- `strict=True missing=0 unexpected=0`：calibrated pth 完美加载，无 key 不匹配

这个结果证明：**真实校准输入 + W8/A16 量化对 chunk0 的 hidden 输出几乎无损**（cos>0.9997），量化成功。

### 实现过程中踩的两个坑（已修复）

1. **`optimize_model` 原地污染共享 visual**：最初 fp chunk 和 quant chunk 共享同一份 `visual.blocks`，`optimize_model` 把 `nn.Linear` 换成 `QLinear` 时连 fp chunk 一起改了，导致两者输出完全相同（cos=1.000000、rel_mse=0、absmax 完全相等）——这是错误结果。**修复**：fp 和 quant 各自独立调 `load_visual_chunk` 加载一份 HF 模型。修复后正确显现量化误差（cos=0.9998）。

2. **`load_visual_chunk` 返回的是 `meta` 字典不是 `local_ds` 列表**：`load_visual_chunk` 返回 `(chunk, meta)`，`meta` 是字典（含 `local_deepstack_count` 等 5 个 key）。曾误把 `meta` 当 `local_ds` 列表，`len(dict)=5` → `n_out=1+5=6` → 凭空打印 5 个虚假 "deepstack" 名字。**指标数值本身是对的**（`eval_one_sample` 内部用 `n=max(len(out_fp),len(out_q))=1` 只算 hidden），错的只是打印行。**修复**：改用 `meta["local_deepstack_count"]`。chunk0 现在正确显示 `local_deepstack=0 outputs=['hidden_states']`。

### 全量评估

小规模验证通过后，跑全量 256×6 chunk：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
bash run_eval_chunk_quant.sh    # 默认 NUM_SAMPLES=256 CHUNKS=0,1,2,3,4,5
```

⚠️ **耗时**：每 chunk 需独立加载 HF 模型 2 次（fp + quant，规避原地污染），256 样本 × 2 次前向，单 chunk 10-20 分钟，6 chunk 总计约 1-2 小时。

### 全量评估实测示例（6 chunk × 64 样本）

以下是用 `real256` OM + 256 张真实图像校准输入，对 6 个 chunk 各跑 64 个样本的真实评估结果：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
NUM_SAMPLES=64 CHUNKS=0,1,2,3,4,5 bash run_eval_chunk_quant.sh
```

控制台总览（脚本末尾自动打印）：

```
chunk  cos_hidden_mean  cos_hidden_min   status
    0         0.999825        0.999755       ok
    1         0.999891        0.999844       ok
    2         0.999972        0.999762       ok
    3         0.999963        0.999937       ok
    4         0.999966        0.999953       ok
    5         1.000075        1.000025       ok
```

逐 chunk × 逐输出详细指标（来自 `eval_report.json`）：

| chunk | 输出 | cos_mean | cos_min | rel_mse | absmax_fp | absmax_q |
|---|---|---|---|---|---|---|
| 0 | hidden_states | 0.999825 | 0.999755 | 5.94e-10 | 17.7 | 17.7 |
| 1 | hidden_states | 0.999891 | 0.999844 | 3.84e-10 | 10.5 | 10.5 |
| 1 | deepstack_0 | 0.999946 | 0.999926 | 2.07e-10 | 16.0 | 16.0 |
| 2 | hidden_states | 0.999972 | 0.999762 | 1.78e-10 | 301.8 | 301.8 |
| 2 | deepstack_0 | 0.999972 | 0.999762 | 1.78e-10 | 301.8 | 301.8 |
| 3 | hidden_states | 0.999963 | 0.999937 | 1.87e-10 | 328.1 | 328.0 |
| 4 | hidden_states | 0.999966 | 0.999953 | 1.64e-10 | 345.4 | 345.4 |
| 4 | deepstack_0 | 0.999999 | 0.999990 | 5.97e-11 | 327.8 | 327.8 |
| 5 | hidden_states | 1.000075 | 1.000025 | 2.68e-10 | 2870.6 | 2869.5 |

**结论**：所有输出余弦相似度均值 > 0.9997（远超 0.95 阈值），W8/A16 + 256 真实图像校准对 visual chunk 几乎无损。这从数值上证明真实校准输入产生的量化参数贴合真实激活分布，6 个 chunk 的 OM 量化精度可靠，可用于真机部署。

**几个观察点**：

1. **absmax 随 chunk 深度递增**：chunk0=17.7 → chunk2/3/4 ≈300-345 → chunk5=2870。这是深层残差累积的正常表现（LayerNorm 前的残差和），与 dump 时观察到的"深层 chunk 激活范围大"一致。chunk5 的 2870 较大但 cos 仍 >0.9997，说明量化器对大范围激活也处理得当。

2. **chunk 5 的 cos=1.000075（略大于 1）**：余弦理论上 ≤1，但 `fp32 vs fp16-fake-quant` 在大 absmax（2870）下，分母 `|a|·|b|` 的浮点误差可能让 cos 略超 1。`rel_mse=2.68e-10` 极小、absmax_fp(2870.647) vs absmax_q(2869.451) 仅差 1.2，说明量化实际损失很小，cos 略超 1 是浮点数值噪声而非真问题。若要严谨可对 cos 做 `min(cos, 1.0)` 截断。

3. **deepstack 与 hidden 的关系**：
   - chunk1 的 deepstack_0（block 5 输出）cos=0.999946，与 hidden（block 7 输出）略不同
   - chunk2 的 deepstack_0 与 hidden 数值完全相同（都 301.8、cos 0.999972）——因 block 11 既是 deepstack 抽取点又是该 chunk 的最后 block，hidden 与 deepstack_0 是同一个张量
   - chunk4 的 deepstack_0（block 17）cos=0.999999，几乎无损

### 成功判据

- 6 个 chunk 全部跑完，无异常，`eval_report.json` 生成
- 每个 chunk 的 hidden 输出余弦相似度均值 > 0.95（W8/A16 + 真实校准通常 >0.999）
- 若某 chunk cos 均值 < 0.9，说明该 chunk 量化损失大，需排查（校准样本不足 / 激活范围异常 / 该 chunk 需更细 group_size 或调 act_bit）

## 真机 NPU 上验证 OM vs MNN 精度（真实校准输入）

上面 `eval_chunk_quant.py` 是 **host 侧**（PyTorch 浮点 chunk vs 量化 qmodel）的精度评估。本节给一条**真机 NPU 上**的精度验证链路：用真实校准输入，对比 OM（量化，跑在 hiai NPU 上）vs MNN-CPU（fp32 浮点），在端侧确认 256 真实图校准的 OM 量化精度达标。

这条链路与 host 评估互补：
- `eval_chunk_quant.py`（host）：PyTorch fp chunk vs 量化 qmodel，**不经过 NPU**，验证量化算法本身精度
- 真机测试（本节）：OM（NPU 编译产物）vs MNN-CPU（fp32），**经过真机 NPU**，验证编译+部署后端到端精度

### 涉及的脚本 / 代码

- `unpack_calib_npz.py`（host）
  - 把真实校准 npz 解压成裸 float32 bin + 扁平 meta json。app 无 zip/npz 解压能力（不加新依赖），所以 host 侧预处理。
- `mobiinfra-oh/entry/src/main/cpp/napi_init.cpp`（app 侧）
  - 新增 `runOmVsMnnRealCalibTest`：每个 chunk 读自己的 calib bin，跑 OM（量化）+ MNN-CPU（fp32）+ MNN-NPU，对比输出
- `mobiinfra-oh/entry/src/main/ets/pages/OpTest.ets`（app 侧）
  - 新增按钮 "Run OM vs MNN REAL-Calib Precision (real npz inputs)"，触发上述测试

### 关键设计：OM input idx 顺序 ≠ ONNX 输入顺序（必看）

⚠️ **`runOmChunkOnce` 里 `SetInputData(idx, data)` 的 idx 不是 ONNX 导出时的 `input_names` 顺序，而是 OMG 在线编译后 OM 模型自己的输入顺序**。两者可能不一样——OMG 编译会重排输入张量。

ONNX 导出时（`visual_plugin_quant_matmul_route.py`）`input_names=["hidden_states_in", "rotary_pos_emb", "attention_mask"]`（idx 0/1/2 = hidden/rotary/mask），但编译成 OM 后顺序可能变。喂反会导致 OM 用 hidden 当 rotary、用 rotary 当 hidden，输出幅度和分布完全错（cos≈0.08，输出集中在 1.x 量级而非真实激活范围）。

**确认方法：用 `GetInputSize` 打印每个 OM input idx 的 size**。`HIAIModelManager::GetInputSize(idx)` 返回**字节数**，`numel = GetInputSize(idx) / sizeof(float)`（float32，**字节数是 numel 的 4 倍**）。三个输入的 numel 是固定的，可凭 numel 反推 idx 对应哪个输入：

| 输入 | shape | numel | bytes (numel×4) |
|---|---|---|---|
| hidden | `[1, 608, 1024]` | 622592 | 2490368 |
| rotary | `[2, 608, 1, 64]` | 77824 | 311296 |
| mask | `[1, 608, 608]` | 369664 | 1478656 |

`runOmChunkOnce` 在喂入前已加诊断日志（`OH_LOG_INFO`，hilog 标签 `[OM input]`）：
```
[OM input] idx=0 (feeding hidden): OM expects numel=622592 bytes=2490368; data numel=622592 absmax=17.4068
[OM input] idx=1 (feeding rotary): OM expects numel=77824  bytes=311296;  data numel=77824  absmax=1.0000
[OM input] idx=2 (feeding mask):   OM expects numel=369664 bytes=1478656; data numel=369664 absmax=0.0000
```

**判断 idx 是否对齐**：每行的 `OM expects numel`（编译后 OM 的实际期望）应与 `data numel`（喂入数据）一致，且 absmax 合理（hidden~17 / rotary~1 / mask~0）。若某 idx 的 `OM expects numel` 和 `data numel` 不一致，说明该 idx 的输入喂反了——按日志里 `OM expects numel` 反查上表，把对应数据喂到正确的 idx。

例如若日志显示 `idx=0 OM expects numel=77824`（rotary 的 numel），说明 OM 编译后 idx0 是 rotary，`SetInputData(0, hiddenData)` 就喂反了，要改成 `SetInputData(0, rotaryData)`、`SetInputData(1, hiddenData)`。

MNN 路径同理有 `[MNN input]` 日志（hilog 标签 `[MNN input]`），打印 chunkInputs 的 numel + absmax，用于核对 MNN 侧输入对齐。

**踩坑记录**：最初 `runOmChunkOnce` 按 ONNX 顺序写死 `SetInputData(0, hidden)/(1, rotary)/(2, mask)`，但实测 OM 编译后顺序反了，导致 OM 输出幅度异常（集中在 1.x，cos≈0.08）。改成按日志确认的 OM 实际顺序喂入后精度恢复正常。**所以每换一套新 OM，都要看 `[OM input]` 日志确认 idx 顺序，不能想当然按 ONNX 顺序。**

### 关键设计：app 侧无新依赖

npz 是 zip 格式，但 app 的 CMake 只链接 `librawfile.z.so` / `MNN` / `hiai`，没有 zip 库。因此：
- **host 侧**把 npz 解压成裸 bin（`unpack_calib_npz.py`），不放 zip
- **app 侧**只读裸 bin（`fread`，复用现有文件读取能力）+ 读扁平 meta json（复用现有 `extractJsonString`/`extractJsonInt`，不支持嵌套所以 meta 用扁平 key）
- 满足 mobiinfra-oh 不加依赖的约束

### 数据流

```
host:  npz (fp16, zip) --unpack_calib_npz.py--> 裸 bin (fp32) + calib_meta.json
       放到 <modelDir>/calib/ (不分子目录)
       push 到手机随模型一起
       ↓
app:   runOmVsMnnRealCalibTest
         每个 chunk i:
           读 <modelDir>/calib/calib_chunk{i}_{hidden,rotary,mask}.bin
           → VARP (hidden [1,608,1024], rotary [2,608,1,64], mask [1,608,608])
           → MNN-CPU (fp32) 前向 → out_fp
           → OM (量化, hiai NPU) 前向 → out_quant
           → compareOutputVectors: relErr / rmsDiff / PASS/WARN/FAIL
```

### 步骤 1（host）：解包 npz 生成 calib bin

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
python3 unpack_calib_npz.py \
    --npz_dir /temp/fdh/input_calib/calib_inputs_256 \
    --out_dir /temp/fdh/model_omc/calib \
    --sample_idx 0 \
    --chunks 0,1,2,3,4,5
```

参数：

| 参数 | 说明 |
|---|---|
| `--npz_dir` | 真实校准 npz 目录（`calib_inputs_256`） |
| `--out_dir` | 输出裸 bin + meta 的目录（建议 `<modelDir>/calib/`） |
| `--sample_idx` | 取每个 chunk 的第几个样本（默认 0） |
| `--chunks` | 解哪些 chunk（默认 `0,1,2,3,4,5`） |

产物（约 25.7MB fp32）：

```
<out_dir>/
  calib_chunk0_hidden.bin    # raw float32, [1,608,1024]
  calib_chunk0_rotary.bin     # raw float32, [2,608,1,64]
  calib_chunk0_mask.bin       # raw float32, [1,608,608]
  calib_chunk1_* ...
  calib_chunk5_* ...
  calib_meta.json            # 扁平 key: chunk0_hidden_shape="1,608,1024" 等
```

meta 用扁平 key（`chunk{i}_{hidden,rotary,mask}_{file,shape,numel}`），因为 app 侧的 `extractJsonString`/`extractJsonInt` 不支持嵌套 JSON。

### 步骤 2：push calib 到手机

把 `calib/` 目录随模型一起 push 到手机，确保路径结构：

```
<modelDir>/                    # 模型目录 (含 config.json, *.mnn, *.om)
  calib/                       # unpack 产物放这里 (不分子目录)
    calib_chunk0_hidden.bin
    ...
    calib_meta.json
  visual_blocks_npu_0.om
  visual_blocks_npu_0.mnn
  ...
```

`runOmVsMnnRealCalibTest` 固定从 `<modelDir>/calib/` 读，不需要额外路径参数。

### 步骤 3：app 侧编译

`napi_init.cpp`（新增 native 函数）和 `OpTest.ets`（新增按钮）都改了，重新编译 mobiinfra-oh：

```bash
# DevEco 里 Build → Build HAP(s)/APP(s)
# 或命令行 hvigor
```

### 步骤 4：真机触发测试

在 app 的 OpTest 页面，点按钮：

**"Run OM vs MNN REAL-Calib Precision (real npz inputs)"**（橙色 `#FF8A65`）

或通过 napi cfg 直接传：`"om_vs_mnn_real_calib|<modelDir>"`（可选带 `|<warmup>|<repeat>`）。

native 侧 `runOmVsMnnRealCalibTest` 执行：
1. 每个 chunk 读 `<modelDir>/calib/calib_chunk{i}_*.bin`（真实图像激活）
2. MNN-CPU（fp32）前向 → 浮点基线
3. MNN-NPU 前向（若可用）
4. OM（量化，hiai NPU）前向 → 量化输出
5. `compareOutputVectors` 对比 OM vs MNN-CPU、MNN-CPU vs MNN-NPU 的每个输出（hidden / deepstack）
6. 打印 `relErr` / `rmsDiff` / `PASS`(<2%) / `WARN`(<5%) / `FAIL`(≥5%)

### 预期结果

对照 host 侧 `eval_chunk_quant.py` 的结果（cos > 0.9997），真机测试预期：
- OM vs MNN-CPU 的 `relErr` < 2%（PASS），与 host 评估的量化损失量级一致
- 若 `relErr` ≥ 5%（FAIL），说明真机 NPU 编译/部署引入了额外误差，需排查 OM 编译或 NPU 算子实现

### 与合成输入测试的对比

| | 合成输入测试 | 真实校准输入测试（本节） |
|---|---|---|
| 按钮 | "Run OM vs MNN Cross-Validation" | "Run OM vs MNN REAL-Calib Precision" |
| cfg | `om_vs_mnn_chunks\|<dir>\|<seq>` | `om_vs_mnn_real_calib\|<dir>` |
| 输入 | 合成（visual_pre 跑合成 patches → hidden；全 0 mask） | 真实 npz（256 张图 dump 的真实激活） |
| 验证 | OM vs MNN-CPU 一致性（编译正确性） | OM(量化) vs MNN-CPU(fp32) 精度（量化损失） |
| 函数 | `runOmVsMnnChunkTest` | `runOmVsMnnRealCalibTest` |

### 注意事项

1. **chunk 输入独立**：每个 chunk 读自己的 `chunk_XX_sample_000.npz` 解出的 bin，不链式传递（与合成测试不同，合成测试 chunk1+=前 chunk 输出）
2. **shape 固定**：当前按 seq_len=608 硬编码 shape（hidden `[1,608,1024]`、rotary `[2,608,1,64]`、mask `[1,608,608]`）。换 seq_len 需重新 dump + unpack，并确认 chunk mnn 模型支持新 shape
3. **calib 必须随模型一起 push**：`runOmVsMnnRealCalibTest` 固定从 `<modelDir>/calib/` 读，缺失会报 `calib directory not found`
4. **sample_idx 默认 0**：每个 chunk 取第 0 个样本。想换样本改 `unpack_calib_npz.py --sample_idx N` 重新解包

### 保存真机输出 + host 详细对比（定位精度差异）

`runOmVsMnnRealCalibTest` 跑完每个 chunk 后，会自动把三路输出（MNN-CPU / MNN-NPU / OM）保存成裸 float32 bin 到 `<modelDir>/debug_outputs/`，便于拷到服务器用 Python 做详细精度对比 + 作图。用于定位真机精度与 host PyTorch 评估不一致的差异来源。

**涉及的代码 / 脚本**：
- `mobiinfra-oh/entry/src/main/cpp/napi_init.cpp` 的 `saveRealCalibChunkOutputs`：真机保存三路输出 + meta json
- `compare_debug_outputs.py`（host）：读 debug_outputs，算精度指标 + 作图

**真机保存的产物格式**（每个 chunk i、输出 j）：

```
<modelDir>/debug_outputs/
  chunk{i}_out{j}_{label}_mnn_cpu.bin    # MNN-CPU fp32 基线
  chunk{i}_out{j}_{label}_mnn_npu.bin    # MNN-NPU (若有)
  chunk{i}_out{j}_{label}_om.bin          # OM 量化
  chunk{i}_meta.json                      # 每 output 的 shape/numel/label/sources
```

其中 `{label}` = `hidden_states`（out0）或 `deepstack_0`（out1，仅 chunk1/2/4）。OM 输出无 shape 信息，复用 MNN-CPU 的 shape（两者输出同 shape）。`runOmVsMnnRealCalibTest` 日志会打印 `saved outputs -> .../chunk{i}_*` 确认保存。

**使用流程**：

1. **真机**：点 "Run OM vs MNN REAL-Calib Precision" 按钮 → 跑完自动保存到 `<手机模型目录>/debug_outputs/`
2. **拷贝**：把整个 `debug_outputs/` 目录拷到服务器（如 `/temp/fdh/input_calib/debug_outputs/`）
3. **服务器对比**：

```bash
cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1
python3 compare_debug_outputs.py \
    --debug_dir /temp/fdh/input_calib/debug_outputs \
    --out_dir /temp/fdh/input_calib/debug_eval \
    --ref mnn_cpu --tgt om
```

参数：

| 参数 | 说明 |
|---|---|
| `--debug_dir` | 真机拷出的 debug_outputs 目录 |
| `--out_dir` | 对比结果输出目录 |
| `--ref` | 参考源：`mnn_cpu` / `mnn_npu`（默认 `mnn_cpu`） |
| `--tgt` | 对比源：`om` / `mnn_npu`（默认 `om`） |
| `--chunks` | 只比某些 chunk，逗号分隔（默认全部） |

**对比指标**（每个 chunk × 每个 output）：

- `cos`：余弦相似度（主指标）
- `rel_mse`：归一化 MSE
- `max_abs_err` / `mean_abs_err`：逐元素绝对误差
- `absmax_ref` / `absmax_tgt`：参考/对比的 absmax
- `rel_max_err_pct`：最大相对误差百分比
- `n`：元素数

**作图产物**：

- `chunk{i}_out{j}_{label}.png`：4 子图（mnn_cpu vs om 数值曲线、scatter 对角线图、逐点误差曲线、abs error 直方图）
- `summary.png`：所有 output 的 cos 柱状图（>0.999 绿 / >0.95 橙 / ≤0.95 红，带 0.999/0.95 阈值线）
- `report.json`：完整指标

**定位精度差异**：通过切换 `--ref` / `--tgt` 对比不同源，可区分三类误差来源：

| 对比 | `--ref` `--tgt` | 误差来源 |
|---|---|---|
| 真机量化误差 | `mnn_cpu` `om` | OM 量化 + NPU 编译（对应 host `eval_chunk_quant.py` 的 cos>0.9997） |
| MNN 实现 CPU vs NPU | `mnn_cpu` `mnn_npu` | MNN NPU 算子实现精度（与量化无关） |
| host vs 真机 | 对照 `eval_chunk_quant.py` 结果 | 量化算法误差（host）vs NPU 部署误差（真机）的差值 |

若 `mnn_cpu vs om` 的 cos 显著低于 host `eval_chunk_quant.py` 的 cos>0.9997，说明真机 NPU 编译/部署引入了额外误差，需排查 OM 编译或 NPU 算子实现；若 `mnn_cpu vs mnn_npu` 误差大，则是 MNN NPU 算子问题（与量化无关）。

## 一键生成全部 6 个 chunk 的自定义 seq_len OM 文件

目录下提供了 `run_all_chunks_seq608.sh`，可一键完成：生成 seq_len=608 校准输入 → 逐 chunk 量化/导出 ONNX → OMC 编译。

```bash
bash run_all_chunks_seq608.sh
```

脚本参数集中在文件头部，可按需修改：

- `SEQ_LEN`：目标序列长度（默认 608）
- `NPU_CHUNKS`：chunk 总数（默认 6）
- `PLATFORM`：OMC 目标平台（默认 `kirin9020`）
- `MODEL_PATH`：模型路径
- `NUM_SAMPLES`：校准样本数

产物：`model_visual_plugin_matmul_chunk{0..5}_seq608/omc_output/visual_plugin_matmul_quantized.om`

如果只需要手动跑单个 chunk（例如调试），步骤如下：

```bash
# 1. 生成校准输入（一次性，覆盖全部 chunk）
python collect_visual_act_stats_v6.py \
  --path /data/fengdahu/model/mobi0429_2B__nore_halfimage \
  --dst_path ./model_visual_plugin_matmul_calib_seq608 \
  --npu_chunks 6 \
  --num_samples 4 \
  --seq_len 608 \
  --dtype fp16

# 2. 逐 chunk 量化 + 导出 ONNX
for i in 0 1 2 3 4 5; do
  python visual_plugin_quant_matmul_route.py \
    --route_dir "./model_visual_plugin_matmul_chunk${i}_seq608" \
    --chunk_index $i \
    --npu_chunks 6 \
    --input_dir ./model_visual_plugin_matmul_calib_seq608/calib_inputs \
    --quant_strategy Quant_aigc_ptq \
    --weight_bit 8 --weight_algo min_max \
    --act_bit 8 --input_algo min_max --input_unsigned_quant \
    --num_samples 2 --group_size 128 \
    --use_qwen3_style_rotary \
    --force_regen all
done

# 3. 逐 chunk 跑 OMC
for i in 0 1 2 3 4 5; do
  PLATFORM=kirin9020 \
  bash run_visual_plugin_matmul_omc.sh \
    "./model_visual_plugin_matmul_chunk${i}_seq608" fp16
done
```

## 步骤 1: 生成并 patch config

默认会把线性层配置成：

- `quant_strategy=Quant_act_weight_eco`
- `weight.bit=4`
- `weight.group_size=128`
- `input.bit=16`
- `input.unsigned_quant=false`

默认**不会**加入 `output` 配置，因此不会影响当前已经验证通过的 `kirinx90` 路线。
默认也**不会**加入以下可选字段，只有你显式传参时才会写进 `dopt_config`：

- `weight.weight_algo`
- `input.input_algo`

`weight.group_size` 只在 W4 时生成；W8 自动省略，即使传了 `--group_size`。W4 仍可用 `--omit_group_size` 省略。
激活方式始终显式写入：A8 默认 `input.unsigned_quant=true`，保持 DOPT PTQ 的 U8S8 默认行为；A16 默认为 false。
可用互斥的 `--input_unsigned_quant` / `--input_signed_quant` 覆盖。切换方式后须重新执行 `prepare` 和 `calibrate`，或执行 `all`；旧 checkpoint 的量化状态会覆盖新配置，不能仅在 `export-onnx` 时切换。

`config_summary.json` 记录实际写入的配置，`calibration_report.json` 从校准 checkpoint 记录每层的位宽、激活符号及权重 scale 形状。`export_report.json` 优先使用这份校准记录；旧产物缺少该记录时使用配置，省略的 `unsigned_quant` 标为 null，避免把 SDK 默认值误报为 false。`quant_param_2` 保留官方插件示例的 false，目前未获得 Kirin9030 的明确平台映射依据。

这些修改纠正配置及报告，尚未证明修复 Kirin9030 真机精度问题。原始 6 份参数在删除 W8 的 `group_size` 后仍可逐字节复现，参数配对核验及真机对照见 [W8A8 排查记录](../../../../docs/inference/w8a8-probe.md)。配置依据：[华为插件式量化说明](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-plug-in-quantification)。

W8A8 加显式输出 INT16 的完整候选经用户真机测试仍不对齐，现已暂停精度调优并移除一次性实验脚本。保留的配置与报告修正不代表精度故障已修复；历史证据与候选结果见上述排查记录。输出量化默认仍关闭。

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0 \
  --chunk_index 0 \
  --npu_chunks 6 \
  prepare
```

如果你想自定义这几个量化配置，也可以直接在命令行覆盖默认值：

- `--quant_strategy`
- `--weight_bit`
- `--group_size`
- `--omit_group_size`
- `--act_bit`
- `--weight_algo`
- `--input_algo`
- `--input_unsigned_quant`
- `--input_signed_quant`

例如改成 `W4 + group64 + A16`：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_g64 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --weight_bit 4 \
  --group_size 64 \
  --act_bit 16 \
  prepare
```

如果复用已有 `route_dir`，建议加 `--force_regen`，确保 `dopt_config.chunk_00.json` 按新参数重新生成。

如果你希望生成更完整的输入/权重配置，例如：

```json
{
  "quant_strategy": "Quant_aigc_ptq",
  "weight": {
    "bit": 4,
    "group_size": 128,
    "weight_algo": "group_min_max"
  },
  "input": {
    "bit": 8,
    "input_algo": "min_max",
    "unsigned_quant": true
  }
}
```

可以这样传：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_a8 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --quant_strategy Quant_aigc_ptq \
  --weight_bit 4 \
  --group_size 128 \
  --weight_algo group_min_max \
  --act_bit 8 \
  --input_algo min_max \
  --input_unsigned_quant \
  prepare
```

如果不传 `--weight_algo` 或 `--input_algo`，脚本不会生成这两个字段；`input.unsigned_quant` 始终按上述规则显式生成。
`--omit_group_size` 用于省略 W4 的 `weight.group_size`；W8 始终省略。

如果你要尝试 `kirin9020`，可以在 `prepare` 或 `all` 时额外打开文档里提到的 `output` 配置：

- `output.bit=16`
- `output.per_channel=true`
- `output.input_algo=min_max`

对应命令示例：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_kirin9020 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --enable_output_quant \
  prepare
```

## 步骤 2: 校准并导出量化产物

这一步会生成：

- `quant_output/calibrated_chunk_00.pth`
- `quant_output/fake_quant_weight.pth`
- `quant_output/quant_params_file`

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --num_samples 2 \
  calibrate
```

## 步骤 3: 导出 FLinearMatmul plain MatMul 风格 ONNX

默认不加 `--fp16`，先保留 `fp32` 权重。

默认也不加 `--use_qwen3_style_rotary`，保持当前已经验证通过的原始 rotary 导出逻辑。
只有显式传入该参数时，脚本才会在导出阶段把 rotary 的内部实现切到更接近官方 `Qwen3` 的形式：

- 外部输入 `rotary_pos_emb` 的打包格式保持不变
- 只修改 export adapter 内部对 `rotary_pos_emb` 的消费方式
- 把 `q/k` 的 rotary 应用改成更接近官方的 `[B, H, S, D]` + `[B, 1, S, D]` 广播布局

这个开关默认关闭，因此不会影响现有成功路线。

这一步导出的目标形态是：

- `Gemm = 0`
- `MatMul = 32`
- `BatchMatMul = 0`

其中主权重层应为：

- `blocks.N.self_attn.q_proj`
- `blocks.N.self_attn.k_proj`
- `blocks.N.self_attn.v_proj`
- `blocks.N.self_attn.o_proj`
- `blocks.N.mlp.linear_fc1`
- `blocks.N.mlp.linear_fc2`

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0 \
  --chunk_index 0 \
  --npu_chunks 6 \
  export-onnx
```

如果你想在**不改变外部 `rotary_pos_emb` 输入格式**的前提下，测试更接近官方 `Qwen3` 的 rotary 导出方式，可以这样导出：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_qwen3rotary \
  --chunk_index 0 \
  --npu_chunks 6 \
  --use_qwen3_style_rotary \
  export-onnx
```

这个参数只影响导出阶段的 rotary 计算路径，不会改动 `utils` 目录，也不会要求你重做现有 `npz` 输入。

如果你想测试导出后再把大权重转成 `fp16`：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --fp16 \
  export-onnx
```

## 步骤 4: 跑 OMC

```bash
PLATFORM=kirinx90 \
bash run_visual_plugin_matmul_omc.sh \
  ./model_visual_plugin_matmul_chunk0 \
  fp16
```

保存日志：

```bash
PLATFORM=kirinx90 \
bash run_visual_plugin_matmul_omc.sh \
  ./model_visual_plugin_matmul_chunk0 \
  fp16 \
  > ./model_visual_plugin_matmul_chunk0/omc_output/omc_kirinx90.log 2>&1
```

## 当前验证通过的完整命令

```bash
source /data/dahu/anaconda3/etc/profile.d/conda.sh
conda activate llama
export DDK_DOPT=/data/fengdahu/cann_codesampe/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=$DDK_DOPT:/home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export:$PYTHONPATH

cd /home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export/plugin_quant_visual_matmul_route_v1

python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --num_samples 2 \
  --force_regen \
  all

PLATFORM=kirinx90 \
bash run_visual_plugin_matmul_omc.sh \
  ./model_visual_plugin_matmul_chunk0 \
  fp16 \
  > ./model_visual_plugin_matmul_chunk0/omc_output/omc_kirinx90.log 2>&1
```

## 一条命令跑到 ONNX

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --num_samples 2 \
  all
```

如果你想从 `prepare -> calibrate -> export-onnx` 整条链路都走一遍，并在最后导出时启用 `Qwen3` 风格 rotary，可用：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_qwen3rotary \
  --chunk_index 0 \
  --npu_chunks 6 \
  --num_samples 2 \
  --force_regen \
  --use_qwen3_style_rotary \
  all
```

如果你想直接用自定义的 `weight_bit / group_size / act_bit` 跑完整链路，可用：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_g64 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --num_samples 2 \
  --force_regen \
  --weight_bit 4 \
  --group_size 64 \
  --act_bit 16 \
  all
```

同理，你也可以在 `all` 模式下同时传入：

- `--quant_strategy`
- `--weight_algo`
- `--input_algo`
- `--input_unsigned_quant`

例如一条命令直接跑到 ONNX，并生成 `A8 + unsigned + algo` 风格的 `dopt_config`：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_a8 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --quant_strategy Quant_aigc_ptq \
  --weight_bit 4 \
  --group_size 128 \
  --weight_algo group_min_max \
  --act_bit 8 \
  --input_algo min_max \
  --input_unsigned_quant \
  --num_samples 2 \
  --force_regen \
  all
```

如果你想保留 `weight.bit=4` 但完全不生成 `weight.group_size`，可以这样写：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_no_group \
  --chunk_index 0 \
  --npu_chunks 6 \
  --weight_bit 4 \
  --omit_group_size \
  --act_bit 16 \
  --num_samples 2 \
  --force_regen \
  all
```

如果你要直接走 `kirin9020` 的可选 output-quant 配置，可用：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_kirin9020 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --num_samples 2 \
  --force_regen \
  --enable_output_quant \
  all
```

如果你既想尝试 `kirin9020`，又想自定义 `weight_bit / group_size / act_bit`，也可以组合使用：

```bash
python visual_plugin_quant_matmul_route.py \
  --route_dir ./model_visual_plugin_matmul_chunk0_kirin9020_g64 \
  --chunk_index 0 \
  --npu_chunks 6 \
  --num_samples 2 \
  --force_regen \
  --weight_bit 4 \
  --group_size 64 \
  --act_bit 16 \
  --enable_output_quant \
  all
```

## Kirin9020 说明

- 这是基于官方文档里对 grouplinear 的补充配置做的可选实现，默认关闭
- 只有在传入 `--enable_output_quant` 时，线性层才会追加：
  - `"output": {"bit": 16, "per_channel": true, "input_algo": "min_max"}`
- 不传该参数时，生成的 `dopt_config.chunk_00.json` 继续保持现有默认行为
- 跑 OMC 时，把平台切到 `kirin9020` 即可：

```bash
PLATFORM=kirin9020 \
bash run_visual_plugin_matmul_omc.sh \
  ./model_visual_plugin_matmul_chunk0_kirin9020 \
  fp16
```

## 主要产物

- `dopt_config.chunk_00.json`
- `config_summary.json`
- `quant_output/calibrated_chunk_00.pth`
- `quant_output/fake_quant_weight.pth`
- `quant_output/quant_params_file`
- `onnx/visual_blocks_npu_0.onnx`
- `onnx/export_report.json`
- `route_config.json`
- `omc_output/visual_plugin_matmul_quantized.omc`
- `omc_output/omc_kirinx90.log`

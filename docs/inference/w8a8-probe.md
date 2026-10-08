# Kirin9030 / 路径 A：W8A8 chunk0 真机对照

> W8A8 当前状态：用户下载输出 INT16 的完整候选后，反馈真机输出仍不对齐，现已暂停该路径的精度调优。未取得此次测试的详细数值日志，因此不补造新的误差指标，也不将该候选标为有效修复。

> 后续 W4A16 状态（2026-10-08）：用户测试独立的 ViT W4A16/group64 发布包后反馈能够区分不同图片，暂按图文功能可用记录。配置、产物和证据边界见 [视觉 NPU 记录 §13](visual-npu-notes.md#13-vit-w4a16group64-真机初测可用2026-10-08)。这是 W4A16 发布包的初测结果，不表示本篇 W8A8 数值故障已修复。
>
> 本文保留历次实验的证据和操作记录。一次性实验脚本及其配套测试已归档后从项目移除，下方实验构建命令是历史记录，不再是当前工程入口。正式导出仍保留配置与报告修正；保留的配置回归测试见文末。本次代码清理未修改手机 App 或操作设备。

这套测试用于定位 `compress_conf` 引入的误差。它独立运行 chunk0，所有实验使用相同的路径 A 输入，同时比较两个 CPU 参考：现有 fake-quant 权重 ONNX 的浮点前向，以及同一校准 checkpoint 的 DOPT W8A8 前向。实验图放在独立目录，尚未证明可替换正式模型。

## 已知证据与验证边界

- 现有压缩参数能由校准 checkpoint 逐字节复现，包含 UINT8 激活、INT8 权重、INT32 bias 和输出反量化参数。
- 手机日志显示 6 个 chunk 执行成功且无 CPU fallback；这只能证明执行成功。chunk1 收到的 hidden 最大值 152.75 明显高于路径 A 校准输入范围，优先检查 chunk0 的计算和输出交接。校准样本与手机照片不同，范围比较本身不能定位根因。
- CPU 上启用 A8 后已出现可测的 chunk0 误差。因此同时比较浮点和 W8A8 参考，才能区分量化误差与离线图执行误差。
- 原 ONNX 是两输入 `MatMul` 加独立 bias `Add`，压缩文件却有三个输入参数；另一个疑点是 `quantInfoExt` 未设置 `pch:1`。下面的实验分别检验这些差异，不预设哪个配置正确。
- `shape` 信息日志的实际影响尚未确定，不向正式参数文件猜测性地补 shape。

## 生成测试包（历史脚本）

在 mobiinfer 根目录运行。本机 Python 实际路径以以下命令为准；不需要重新校准或导出整个 LLM。

```bash
export PYTHONPATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3:$PWD/transformers/llm/export
export PYTHONDONTWRITEBYTECODE=1
/home/ma-user/.conda/envs/CANN/bin/python \
  transformers/llm/export/plugin_quant_visual_matmul_route_v1/prepare_w8a8_probe.py \
  --output_dir /temp/work/w8a8_probe --compile --include_gemm
```

输出目录必须不存在；重跑时选择新的目录。默认使用路径 A 的 sample 0、1、10、100。脚本先验证两份控制图的 ONNX 和外部权重一致，再验证 checkpoint 的 W8 浮点激活前向与 ONNX 一致，以及重新生成的压缩参数与原文件 SHA-256 一致。

| case | 唯一实验变化 | 设备输出应比较的参考 |
|---|---|---|
| `float_control` | 现有 fake-quant 权重、无 compress_conf 的控制图 | `float` |
| `w8a8_baseline` | 现有原始 W8A8 图 | `w8a8` |
| `no_bias_param` | 仅移除压缩配置的 input index=2；保留 ONNX 独立 Add | `w8a8` |
| `pch` | 通过 DOPT 官方 serializer 使用 `pch=True` | `w8a8` |
| `gemm` | 将 24 个带常量 bias 的 MatMul+Add 合成三输入 Gemm，保留原参数 | `w8a8` |

Gemm 会先通过 CPU 浮点等价检查。编译实验图时必须得到 OMC、`NPU:1, CPU:0` 和成功日志；失败的 case 记录在 manifest 中，不加入设备测试列表。编译成功仍需真机数值验证。

`/temp/work/w8a8_probe.zip` 仅包含设备运行需要的图、输入、参考、`probe.tsv` 和 `manifest.json`。`build/` 内的编译日志、实验 ONNX 和参数明文不进入 ZIP。manifest 记录运行文件 SHA-256，方便核对传输。

## 本次生成结果（2026-10-07）

已生成 `/temp/work/w8a8_probe.zip`，约 281 MiB。实际可运行的组是 `float_control`、`w8a8_baseline`、`pch`、`gemm`，共 16 次独立前向。

| sample | 浮点参考 absmax | W8A8 参考 absmax | W8A8 vs 浮点 cosine | relative L2 |
|---|---:|---:|---:|---:|
| 0 | 17.2808 | 18.1844 | 0.98187 | 0.19587 |
| 1 | 17.6879 | 19.0249 | 0.96804 | 0.25766 |
| 10 | 17.9782 | 17.4708 | 0.97215 | 0.24086 |
| 100 | 16.1040 | 17.6680 | 0.96330 | 0.27459 |

关闭 A8 时，checkpoint 与 ONNX 的 relative L2 为约 `1.3e-6`–`1.6e-6`。这排除了本次参考生成中权重或前向不匹配的明显问题；启用 A8 后的误差已经很大，尚未验证完整模型的回答质量。

`no_bias_param` 编译失败：删除 input index=2 后，OMG 转入 `LegacyQuantizeSaver`，报 `can not find winoFlag` 和 `save quantize info ext failed`。所以当前不能用简单删除 bias 参数作为修复，也不能把这次失败解释成已经证明了 bias 根因。

`pch` 与原配置的唯一差异已通过加密文件字节比较确认：24 处 `version:2` 改为 `version:2;pch:1`。其编译仍有 24 条 “no bias weight” 和 192 条 “anchor is invalid”；`gemm` 的前者降为 0，后者仍是 192。两者都有 `NPU:1, CPU:0`；下面的真机结果表明这两项变化均未修复故障。

## 真机结果（2026-10-07 16:47）

已通过 HDC 取回 `results/`，保存在 `/temp/work/w8a8_probe/device_results/`。16 份输出均为 622592 个有限 float32 元素，重新逐元素计算的指标与手机 `report.tsv` 一致；报告及代表性输出的 SHA-256 与手机端相同。此次结果是 4/16 对齐、0 个执行/文件错误。

已同时取得 `mnn_runtime.log`：HiAI 版本为 `109.636.120.010`，四组图的三个输入与 hidden 输出描述完全一致，均为 FP32/NCHW；hidden 是 `[1,1,608,1024]`，输入映射为 hidden=0、rotary=1、mask=2。四组逐样本 hidden 输入 hash 相同，16 次 RunSync 均成功，保存的输出 hash 与 `output_values` 日志一致。当前差异不是测试入口的输入顺序、FP16/FP32 误读或输出文件写入造成的。

| case | 对照参考 | cosine 范围 | relative L2 范围 | 设备 absmax 范围 |
|---|---|---:|---:|---:|
| `float_control` | CPU 浮点激活 / fake-quant 权重 | 0.999978–0.999981 | 0.00614–0.00666 | 16.14–17.97 |
| `w8a8_baseline` | CPU DOPT W8A8 | 0.10851–0.12276 | 21.20–22.91 | 153.50–154.25 |
| `pch` | CPU DOPT W8A8 | 与 baseline 完全相同 | 与 baseline 完全相同 | 与 baseline 完全相同 |
| `gemm` | CPU DOPT W8A8 | 0.10850–0.12274 | 21.20–22.91 | 153.50–154.25 |

四个样本的 `pch` 与 baseline 输出文件逐字节相同。Gemm 与 baseline 的 relative L2 仅 0.00269–0.00285，余弦约 0.999996；结构转换有小影响，但没有改善约 21–23 倍的参考误差。因此，单独增加 `pch:1` 或将 MatMul+Add 合并为 Gemm，均不是本次故障的有效修复。“no bias weight” 日志消失也不能证明计算正确。

以 sample 0 对其他样本计算整张 hidden 张量的余弦：浮点控制为 0.536–0.597，baseline 为 0.99794–0.99820。baseline 四个样本的标准差都约 10.7，而浮点控制约 0.38–0.43。不同图片的压缩输出被共同的大幅特征主导，与图文不相关的表现相符；这些输出仍有变化，不能称为完全相同。

这组数据把问题定位到 `compress_conf` 触发的 NPU 量化路径与 DOPT 前向不一致，且在 chunk0 已经出现。普通的 A8 误差只有本次 CPU 对照中的 relative L2 0.20–0.27，不能单独解释设备对量化参考 21–23 的偏差。压缩参数仍可从 checkpoint 精确复现，因此目前没有证据说明参数数值文件损坏；需要检查其在 OMG/Kirin9030 中被消费的方式。

最初计划建立第一层算子级对照。后续按用户要求改为直接验证参数消费兼容性，重点检查激活零点、权重方向、bias 补偿与反量化尺度，不逐层定位首次偏离的算子。

本机还在所有 24 个激活 quantizer 上分别模拟了漏减零点、UINT8→INT8 平移后使用错误零点、零点符号反向、UINT8 按有符号重解释；没有精确复现设备输出。这只能排除这些“所有层都发生同一种简单错误”的模型，不能排除个别算子的零点处理错误，也不能直接宣布某一种错误就是根因。

## 参数兼容性修复实验（2026-10-07）

`repair_w8a8_params.py` 在独立目录生成候选图、参数和 CPU 参考，原校准 checkpoint、原压缩文件及正式 App 模型不变。本轮未进一步修改 App 或推理引擎。

```bash
export PYTHONPATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3:$PWD/transformers/llm/export
export PYTHONDONTWRITEBYTECODE=1
/home/ma-user/.conda/envs/CANN/bin/python -B \
  transformers/llm/export/plugin_quant_visual_matmul_route_v1/repair_w8a8_params.py \
  --compile --output_dir /temp/work/w8a8_params_fix_v2
```

这个目录已生成并执行真机测试。重跑时换用不存在的新目录。上传到手机的 39 个运行文件 SHA-256 全部一致；取回的 24 份输出均通过元素数量及有限值检查，重算指标与 `report.tsv` 一致。结果保存在 `/tmp/w8a8_params_fix_v2_results/`。

| case | 参数/图变化 | 对对应 CPU 参考的 cosine | relative L2 | 设备 absmax |
|---|---|---:|---:|---:|
| `float_control` | 无压缩控制图 | 0.999978–0.999981 | 0.00614–0.00666 | 16.14–17.97 |
| `w8a8_baseline` | 原 UINT8 非零零点 | 0.10851–0.12276 | 21.20–22.91 | 153.50–154.25 |
| `gemm_nt_uint8` | 权重物理转为 `[N,K]`，Gemm `transB=1` | 0.10850–0.12274 | 21.20–22.91 | 153.50–154.25 |
| `matmul_signed_exact` | `UINT8→INT8`，`z_signed=z_unsigned-128`，scale 不变 | 与 baseline 完全相同 | 与 baseline 完全相同 | 与 baseline 完全相同 |
| `matmul_signed_symmetric` | INT8、零点 0，scale=`max(abs(min),abs(max))/127` | 0.95807–0.96410 | 0.26866–0.29221 | 17.52–18.45 |
| `gemm_nt_signed_symmetric` | 标准权重布局，加对称 INT8 | 0.95809–0.96448 | 0.26714–0.29208 | 17.41–18.52 |

INT8 等价表示在四个 CPU 样本上与原 UINT8 输出逐值相同；它在真机上的四个输出也与 baseline 逐字节相同。标准权重布局没有改善。零点为 0 的候选消除了本组输出幅度异常，但仍未达到探针的对齐阈值，**不能宣布 W8A8 修复成功**。

对称 INT8 改变了量化网格，与原 DOPT W8A8 的 CPU relative L2 为 0.32–0.41，对浮点参考为 0.34–0.44；表格的设备指标使用的是新网格的独立 CPU 参考。这个实验同时改变零点和 scale，支持继续检查非零零点路径，尚不能单凭它确定具体错误。

下一步准备两组独立测试，不替换正式模型：

- `--parameter_contract`：使用精确可表示的对角 Linear 权重、非零 bias，分别测试 INT8 零点 0、-7、31 和 UINT8 零点 121、128。CPU 参考由标准 ONNX QuantizeLinear/DequantizeLinear 生成，直接检验参数语义；不依赖寻找 ViT 首个偏移算子。
- `--explicit_zero_point`：保留原激活 scale，把 `z_signed=z_unsigned-128` 显式改写为 `x'=x+s*z_signed`、`b'=b-s*z_signed*sum_K(W)`，压缩参数改用 INT8 零点 0；分别编译 Gemm 内部 bias 和独立 Add bias。另含零点符号取反的参数语义候选。该符号候选不是已证明等价的修复。

显式补偿的浮点数学等价性与量化参考分别校验。FP32 运算顺序和量化边界会改变部分编码：ONNX QDQ 与 DOPT 在本组 CPU 输入上已有约 4%–6% relative L2 差异，因此不能宣称重写后逐位等价。候选以自己的独立 QDQ 参考作为设备执行对照，同时记录与原 DOPT 的差异。

这两组测试包的实际目录、编译状态以 manifest 为准。操控手机前必须先获得用户明确确认；输入测试路径时先清空原内容，再输入完整路径，并读取 UI 控件核对后运行。

已生成 `/temp/work/w8a8_parameter_contract_v2`（约 82 MiB，6 组 × 4 输入）与 `/temp/work/w8a8_explicit_zero_point_v2`（5 组 × 4 输入）。全部 11 个 case 满足 OMC 成功与 `NPU:1, CPU:0`。第一版小图因 ReduceMean 产生标量输出而回退 CPU，已拒绝；v2 保持 `[1,1,1]` 输出，未放宽编译判据。

### 第三轮真机验证：零点语义与显式补偿

用户确认后已完成上述 44 次测试。两套包的 77 个运行文件 SHA-256 与手机端一致；两次路径切换均先全选删除，并通过 UI 控件确认内容为空，再输入、提交并核对完整路径后启动测试。全部输出为 622592 个有限 float32 元素，重算两种参考指标均与 report.tsv 一致；native 日志中有对应 44 次 RunSync success、44 组 hidden 输入和输出统计。

结果保存在 `/tmp/w8a8_parameter_contract_v2_results/`、`/tmp/w8a8_explicit_zero_point_v2_results/`，本轮 runtime 日志是 `/tmp/w8a8_current_runtime.log`。

| 参数小图 | cosine 范围 | relative L2 范围 | 判断 |
|---|---:|---:|---|
| 浮点控制 | 0.999999967–0.999999969 | 0.000659–0.000687 | 对齐 |
| INT8 零点 0 | 0.999999987–0.999999989 | 0.000677–0.000687 | 对齐 |
| INT8 零点 -7 | 0.999985769–0.999988422 | 0.00485–0.00536 | 对齐 |
| INT8 零点 31 | 0.999969807–0.999970762 | 0.00767–0.00779 | 对齐 |
| UINT8 零点 121 | 与 INT8 零点 -7 逐字节一致 | 同前 | 对齐 |
| UINT8 零点 128 | 与 INT8 零点 0 逐字节一致 | 同前 | 对齐 |

24 个小图 CPU 参考还由独立 NumPy 算法重新验证，最大绝对差约 `1.75e-6`。这组使用统一的 weight scale 和精确可表示权重，说明基本的非零零点参数在这个场景中可以正常消费；不能据此排除非均匀通道 scale、极小 scale 或更复杂的融合路径问题。

| ViT case | 对对应量化参考 cosine | relative L2 | 设备 absmax |
|---|---:|---:|---:|
| 原 W8A8 | 0.10851–0.12276 | 21.20–22.91 | 153.50–154.25 |
| 零点符号取反 | -0.03649–-0.01549 | 21.16–22.91 | 134.38–136.38 |
| 显式补偿、Gemm bias | 0.10886–0.12293 | 21.21–22.92 | 153.63–154.38 |
| 显式补偿、独立 Add bias | 0.10844–0.12264 | 21.26–22.98 | 153.88–154.63 |

第二组只有 4 个浮点控制通过，16 个压缩 case 不对齐，均无执行/文件错误。因此，这三种参数或图改写不能作为修复。先前“非零激活零点本身”的推断需要收窄；对称化同时改变 scale 和量化网格，不能仅凭改善就断定零点根因。

### 第四轮真机验证：极小权重 scale 与非均匀通道 scale

原 chunk0 的 `blocks.0.mlp.linear_fc1` 存在 weight scale=`1.4901161193847656e-8` 的通道，bias/输出 scale=`3.738200415881465e-9`。两者转换成 FP16 都成为 0；bias 除以原 scale 的最大绝对值约 `1.44e9`，本次没有发现原 bias 自身超出 INT32 范围。这是潜在的参数精度兼容风险，尚未证明 OMG 确实把该 scale 存成 FP16，不能宣布它就是根因。

已准备 `/temp/work/w8a8_weight_scale_floor`：保持激活参数，设置 weight scale 下限 `2^-14`，只重编码所选输出通道的权重，并同步更新 bias/输出 scale。四层 fc1 分别涉及 2、61、117、249 个通道。四个 CPU 浮点参考的 relative L2 仅 `4.05e-7`–`4.47e-7`；对应 QDQ 量化输出与修改前的 QDQ 相同，仍与原 DOPT 有约 4%–6% 差异。测试包含原浮点控制、原 W8A8 和该候选，共 12 次。

另准备 `/temp/work/w8a8_channel_scale_contract`：小图改用五档非均匀 per-channel scale，并加入一个零权重通道，其 scale=`2^-26`、bias=-5.375，检验基本非零零点与极小/非均匀权重尺度的组合，共 24 次。两套包全部编译为纯 NPU 离线图。

用户确认后，已完成上述 36 次测试。71 个运行文件的手机与本机 SHA-256 一致；输入两套路径时均先全选删除、读取 UI 确认空值，再输入并核对完整路径。结果目录为 `/tmp/w8a8_weight_scale_floor_results/` 和 `/tmp/w8a8_channel_scale_contract_results/`，运行日志为 `/tmp/w8a8_scale_runtime.log`。36 份输出都是 622592 个有限 float32 元素；重新计算的两种参考指标与报告一致。对应 native 日志包含 36 次 RunSync success、36 组 hidden 输入与输出，I/O 描述仍为 FP32/NCHW。

权重尺度下限候选的四份设备输出与原 W8A8 **逐字节一致**，没有改善。其 relative L2 仍为 21.20–22.91、cosine 为 0.10864–0.12283；报告相对原 baseline 的小数值差别来自候选使用 QDQ 参考，不能解释为设备输出改善。12 次中只有 4 次浮点控制对齐，8 次压缩测试均不对齐。

| 非均匀通道尺度小图 | cosine 范围 | relative L2 范围 | 判断 |
|---|---:|---:|---|
| 浮点控制 | 0.999999950–0.999999952 | 0.000597–0.000617 | 对齐 |
| INT8 零点 0 | 0.999999961–0.999999962 | 0.000616–0.000618 | 对齐 |
| INT8 零点 -7 | 0.999963747–0.999966564 | 0.00819–0.00853 | 对齐 |
| INT8 零点 31 | 0.999919273–0.999923894 | 0.01235–0.01271 | 对齐 |
| UINT8 零点 121 | 与 INT8 零点 -7 逐字节一致 | 同前 | 对齐 |
| UINT8 零点 128 | 与 INT8 零点 0 逐字节一致 | 同前 | 对齐 |

全部 24 次通过；独立 NumPy 参考与 ONNX QDQ 参考的最大绝对差为约 `1.77e-6`。极小 scale 通道本身的设备最大误差约 `0.00145`。但该通道权重为零，小图的权重矩阵也是对角矩阵，因此没有完整覆盖稠密权重、非方阵权重打包、大的零点补偿或融合 INT32 bias 的一般情形，不能推广为“真实 ViT 的参数消费一定正确”。

生成命令：

```bash
/home/ma-user/.conda/envs/CANN/bin/python -B \
  transformers/llm/export/plugin_quant_visual_matmul_route_v1/repair_w8a8_params.py \
  --weight_scale_floor 0.00006103515625 --compile \
  --output_dir /temp/work/w8a8_weight_scale_floor

/home/ma-user/.conda/envs/CANN/bin/python -B \
  transformers/llm/export/plugin_quant_visual_matmul_route_v1/repair_w8a8_params.py \
  --parameter_contract --channel_scale_profile tiny --compile \
  --output_dir /temp/work/w8a8_channel_scale_contract
```

使用前面同样的 PYTHONPATH；重跑必须换新的输出目录。权重 scale 下限仅是失败的独立实验选项，不作为正式修复。

### 停止继续试参后的重新分析

按用户要求，停止新增参数候选、编译试验和手机操作。下面只依据已有代码、编译日志和结果，不把新猜测转换成新一轮测试。

1. **项目输入和原始参数的明显错误缺少证据。** 原 chunk0 的 24 条参数都唯一绑定到同名 MatMul，weight scale 数量等于逻辑权重 `[K,N]` 的 N，bias/输出 scale 均等于 `sx * sw`。ONNX fake 权重除以通道 scale 距离整数的最大残差约 `7.18e-6`，整数值在 INT8 范围内。INT32 bias 及加入标准有符号零点补偿后的最大绝对值都约 `1.44e9`，未超过 INT32；这不是对 SDK 实际生成 bias 的验证。App 的离线执行入口不读取 quant_params_file，该文件已在 OMG 编译阶段被消费；修 App 的参数读取不能解决这里的问题。

2. **多数 ONNX 改写没有绕开 SDK 的同一融合路径。** 原编译日志 `/temp/work/model_omc/chunk0_kirin9030pathA/omc_output/omc_kirin9030.log` 最终有 24 个 `QuantBatchMatmulV3`、16 个 `StaticQuant` 和 8 个浮点 attention `BatchMatMulV2`。canonical Gemm 的日志也选择同样的 24 个融合量化 MatMul，且使用相同的 kernel bin 名称；显式零点补偿改写仍选择 `QuantBatchMatmulV3`。这证明内部路径重合，不证明融合权重、bias、scale 或 tiling 数据逐字节相同。日志中的 `LayerNormQuant_pattern` 没有匹配成功，当前不能将 LayerNormQuant 融合作为已发生的根因。

3. **报错样式的警告本身不足以定位。** 对齐的小图同样生成 `StaticQuant` / `QuantBatchMatmulV3`，同样有每个 Linear 对应的 4 条 missing shape、1 条 no bias、8 条 invalid anchor。真实 ViT 的 96/24/192 条数量正好随 24 个 Linear 增加。因此，不应继续仅凭这些警告猜测添加 shape、删除 bias 或修改 pch。

4. **SDK 内部有需要核实的参数转换。** 本地 Kirin9030 的 `npu_ascendc_opinfo.json` 声明 `StaticQuant` 的输入、scale、offset 为 FP16，输出可以为 INT8；`QuantBatchMatmulV3` 使用 FRACTAL_NZ 权重、INT32 bias 和包含 UINT64 编码的 `quant_pre` 等参数。原 DOPT 文件的逻辑 scale/offset 需要在这些表示之间转换。真实 fc2 的 UINT8 零点为 1、2、2、4，等价 INT8 零点为 -127、-126、-126、-124，小图仅覆盖 0、-7、31。仍需核实稠密权重打包后的通道 scale 对应关系、零点补偿 bias、反量化编码，以及融合后的内存/tiling 使用。未取得 SDK 转换后的数值，不能指认其中某一步已出错。

5. **工具链版本配套关系尚未确认。** 本机 DDK 目录为 `next-6.1.1.0`，Kirin9030 插件标记为 `DDK_PLATFORM_PLUGIN_100.600.020.010`，真机 HiAI 为 `109.636.120.010`。这些是不同组件的版本，不能仅因数字不同就判定不兼容；应由华为确认这套 DOPT、OMG、插件、AscendC 和设备 HiAI 是否为受支持的组合。App 已通过模型兼容性检查，但该检查和 RunSync success 不能代替数值一致性验收。

项目侧有依据的改动边界是：严格校验导出的权重/节点/参数绑定，固定工具链组件并记录版本，以及在华为提供支持的接口后，控制具体的量化融合或改用明确支持的量化图表示。这些前两项属于防止配置错误的保护，现有核对已通过，不能称为本次修复。禁用融合、使用 ONNX QDQ 或更换 SDK 也尚未证明受当前 Kirin9030 工具支持或能解决故障，不能直接落入正式导出。当前没有证据支持继续修改 App 张量拷贝、权重轴、pch 或任意调整量化数值。

更有价值的下一步是把现有的 chunk0 数值反例提交华为：同一个 fake 权重 ONNX、同一套固定输入，无 compress_conf 对齐，而 compress_conf 对量化 CPU 参考的 relative L2 为 21–23。要求提供转换后的逻辑 INT8 权重、INT32 bias/零点补偿、StaticQuant scale/offset、Fixpipe 反量化参数，以及版本兼容说明或已知修复。公开 OMG `--mode=1` 对当前 OMC 没有产生 JSON，并报 `there is no subgraph in GraphOp`；不能用旧的其他 IR 模型代替当前 OMC 作为根因证据。本记录尚未发送给外部人员，bug 尚未修复。

## 两篇官方 LLM 文档的核验与输出 INT16 候选（2026-10-07）

已读取 [三段式量化步骤](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-llm-three-stage-quantification) 和 [量化效果评估](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-llm-quantization-effect-evaluation) 的官方正文，分别更新于 2026-07-28、2026-06-05。

评估示例要求在原模型中注入量化算子、严格加载校准 checkpoint，同时启用权重和输入量化，关闭校准后执行 eval。现有 W8A8 探针符合这些步骤，原始 checkpoint 已严格加载；只用 fake_quant_weight 的浮点激活 ONNX 无法代替 A8 仿真参考。示例导入了 set_run_mode，但没有调用它，不能据此推断现有代码少了一次必需调用。本机 SDK 的 API 位于 do_opt，而网页示例从 opt_main 导入，不能不检查版本就照抄路径。

三段式示例使用 LLM 的 Quant_act_weight_eco、W4/group64/A16，并给出逐通道 INT16 输出配置；没有说明这是 ViT W8A8 或 Kirin9030 的必需配置。为检验这个新差异，仅建立一个隔离候选：对原 chunk0 的 24 个 Quant_aigc_ptq Linear 添加 output，冻结原权重和输入量化器，使用原有 256 个真实输入只校准新增输出量化器。没有修改正式导出默认值、原始模型或 App。

`prepare_w8a8_output_probe.py` 生成的主机审计结果位于 `/tmp/w8a8_output_int16/manifest.json`。360 个原始 checkpoint tensor 和 72 个 fake_quant_weight tensor 均逐值保持一致，144 个新增 buffer 均属于输出量化器。官方 serializer 生成的 input/weight/bias 参数及原始输出 sx*sw 参数与原文件一致，仅在输出新增 INT16 Quantize 和 FLOAT AntiQuantize。24 层输出 scale 均为正，通道数为 1024 或 4096，形状为 `[1,N]`；新 checkpoint 严格重载成功，实际加密参数 SHA-256 已改变。

| sample | 新输出 INT16 vs 原 W8A8 relative L2 | 新输出 INT16 vs 浮点 relative L2 | 原 W8A8 vs 浮点 relative L2 |
|---|---:|---:|---:|
| 0 | 0.073541 | 0.196079 | 0.195866 |
| 1 | 0.080854 | 0.257941 | 0.257658 |
| 10 | 0.074746 | 0.238071 | 0.240855 |
| 100 | 0.081586 | 0.274668 | 0.274585 |

新增输出量化在完整 ViT 中产生 7.35%–8.16% 的前向偏移，没有明显改善原有 A8 相对浮点的误差。首次审计时，该候选未通过事先设置的数值保留检查（cosine≥0.999、relative L2≤0.02），脚本完成参数配对审计、保存报告后以非零状态退出，未进入 OMG 编译。这个主机检查失败不等于已经证明真机 INT16 输出失败，也不能证明设备故障已修复。用户随后调整标准后的编译结果见下一节。

另外，本机 Kirin9030 `npu_ascendc_opinfo.json` 的 QuantBatchMatmulV3 声明：INT8/UINT8 输入配 INT8 权重时，直接融合输出仅列 INT8、FP16；INT16 输出组合使用 A16。StaticQuant 和 AntiQuant 各自支持 INT16，但不能据此推断 W8A8 融合 MatMul 可以直接输出 INT16，或编译器会改用正确的新路径。因此，DOPT 接受 output 配置不等于平台支持相应融合组合；不能把 W4A16 示例升级为 W8A8 默认配置。

复现主机审计（首次运行用不存在的新目录，预计需要校准数分钟）：

```bash
export PYTHONPATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3:$PWD/transformers/llm/export
export PYTHONDONTWRITEBYTECODE=1
/home/ma-user/.conda/envs/CANN/bin/python -B \
  transformers/llm/export/plugin_quant_visual_matmul_route_v1/prepare_w8a8_output_probe.py \
  --output_dir /tmp/w8a8_output_int16_recheck
```

对现有隔离目录可加 `--resume_calibration`，严格加载已保存的新 checkpoint 以复核同一个候选，不重新校准；配置或原有权重/输入状态不一致会拒绝复用。13 项相关单元测试通过，其中新增测试覆盖选择性输出校准、正确通道轴、严格重载和非输出状态误改的拒绝。

这两篇文档明确了参考评估和 W4A16 流程，尚未给出解决当前 Kirin9030 ViT W8A8 执行偏差的配置。已有证据仍是：不带 compress_conf 对齐，带原始 compress_conf 相对同一 checkpoint 的官方量化仿真严重不对齐。正式模型和设备测试状态保持原样，bug 尚未修复。

## 用户允许误差后的完整图文候选（2026-10-07）

用户明确允许 NPU/量化误差，要求以图文回答的相关性判断是否可用。因此只放宽数值诊断标准，checkpoint 严格加载、原权重/输入参数保持不变、有限值、文件哈希和纯 NPU 编译检查仍保留。chunk0 将阈值显式设置为 cosine≥0.99、relative L2≤0.10，四个 CPU 样本通过后已编译并打包。之前的 2% 拒绝结果另存 `/tmp/w8a8_output_int16/manifest.strict_gate.json`，没有覆盖历史结论。

首次沙箱内编译同时出现 SDK 工具权限错误和 MatMul unsupported；在完整权限环境重跑，同一个候选得到 `NPU:1, CPU:0`、OMC 和成功日志。因而不能将那次 unsupported 解释为该平台拒绝输出 INT16 整图，也不能仅凭融合算子 dtype 表推断整个图的支持情况。

`prepare_w8a8_output_vlm.py` 已为剩余五个 chunk 使用各自原有的 256 个输入，只校准新增输出量化器。全部六个 chunk 的 2160 个原始 checkpoint tensor、432 个 fake_quant_weight tensor 逐值保留；input/weight/bias 参数及原 sx*sw 输出参数均通过官方 serializer 配对检查。多输出 chunk 的所有 deepstack 参考均保留。

| chunk | 新输出 INT16 vs 原 W8A8 的最小 CPU cosine | 最大 CPU relative L2 |
|---|---:|---:|
| 0 | 0.99667 | 0.08159 |
| 1 | 0.99723 | 0.07449 |
| 2 | 0.99013 | 0.14299 |
| 3 | 0.99941 | 0.03422 |
| 4 | 0.99964 | 0.02701 |
| 5 | 0.99925 | 0.03893 |

这里是新旧 CPU 仿真之间的差异，尚未测量新候选的 NPU 误差。chunk2 超过暂定 10% 诊断标准，如实标记在 manifest；根据用户以图文相关性为准的要求，仍保留在完整候选中，不把它标成数值通过。

六张图均成功编译为纯 NPU OMC，每张约 50 MiB。核心编译算子仍为 24 个 QuantBatchMatmulV3、16 个 StaticQuant 和 8 个 BatchMatMulV2，没有单独 AntiQuant。新旧 OMC 的文件大小相同、SHA-256 不同；这些证据不能证明关键量化计算已改变，更不能证明已修复故障。构建完成时尚未做真机验证；发布后的用户测试仍不对齐，见文首状态。

已生成并校验：

- 固定输入 chunk0 包：`/tmp/w8a8_output_int16.zip`，约 242 MiB，3 组 × 4 个输入。
- 六图视觉替换包：`/tmp/w8a8_output_int16_vlm_ready.zip`，约 298 MiB，包含六个 OM、真实 config 副本、manifest 和说明。它需要与完整模型一起使用。
- 独立完整候选模型目录：`/tmp/w8a8_output_int16_vlm_ready/model/mnn_mobi_2B_w8a8_output_int16`，约 3.35 GiB。本机已复制所需 LLM、tokenizer、embedding、MNN 视觉文件，并使用新六图；没有链接或覆盖源模型。
- 审计记录：`/tmp/w8a8_output_int16_vlm_ready/manifest.json`。ZIP CRC、六图和 config 的 SHA-256 均通过；运行 config 与源 config 相同，deepstack 复制配置仍为 `[2]`。

以下为六图构建的历史命令；对应实验脚本现已归档并移除。当时使用新的输出目录，SDK 初始化/编译需处于能正常访问其工具的环境：

```bash
export PYTHONPATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3:$PWD/transformers/llm/export
export PYTHONDONTWRITEBYTECODE=1
/home/ma-user/.conda/envs/CANN/bin/python -B \
  transformers/llm/export/plugin_quant_visual_matmul_route_v1/prepare_w8a8_output_vlm.py \
  --compile --output_dir /tmp/w8a8_output_int16_vlm_recheck
```

这条脚本生成视觉替换包；独立完整模型由源 runtime 的常规文件副本加上该包的 config/om 组装。校准工作进程在启动前限制数学库线程，避免并行任务过量占用 CPU。14 项相关测试通过，正式 App 和推理引擎未改。

真机图文测试：

1. 使用独立完整候选，或复制现有同源六 chunk 模型到新目录，再替换该副本中的 config.json 和六个 om 文件。保留正常工作的原模型。
2. App 选择该副本，使用离线 NPU 模式。检查日志为 NPU_ACTIVE、graph_ready=6、cpu_fallback=0、npu_error=0。
3. 使用 3–5 张不同场景的图片，每张开启新会话，用同一个问题“这张照片中是什么内容”。与正常 FP16 模型使用相同图片、相同问题，检查主要对象、场景和描述是否符合图片；不要求生成文字逐字相同。
4. 保留每张图片的回答及运行日志。固定输入测试页的 ALIGNED/MISMATCH 仍使用 App 原有 2% 标签，新阈值只用于本机复核；不能仅凭 MISMATCH 标签否定用户允许误差后的图文结果。

上述构建阶段未操作手机。随后用户明确要求直接替换 App 对应离线图，并指定不备份旧图；2026-10-07 已通过 `127.0.0.1:6000` 完成覆盖。目标是 App 当前选中的 `modelscope:fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_exp2_nocompressconf`，其 `om/visual_blocks_npu_0.om` 至 `5.om` 六个手机端 SHA-256 全部与候选一致，`config.json` 哈希仍为 `6c5f3324b581a9e2eb5cc9049d268556554853b34b10cd0768718456b3ed0b39`。视觉前后处理及 `llm_config.json` 在替换前与候选源模型一致；此次仅覆盖六个 OM。

覆盖前已 force-stop App，以清除旧图缓存。该设备 `shell -b` 可读沙箱但拒绝重命名，实际覆盖使用 `file send -b`；本次上传的六个 `.om.upload` 暂存文件已从对应 `/mnt/debugtmp/100/debug_hap/com.coevomind.clawmate/` 调试写入映射清除，没有保留原图副本。审计记录为 `/tmp/w8a8_output_int16_deployment/deployment.json`。模型目录名、旧包说明及旧包 `offline_om_manifest.json`/`SHA256SUMS` 未改，因此旧包名称和旧哈希不再代表当前 OM，当前图以这次审计记录为准。

新图仍是 W8A8 加显式输出 INT16 量化的候选；下一处启用 A8 的 Linear 会再次量化输入。该描述属于导出配置，编译后融合 kernel 内部是否完整保留 INT16 阶段尚未证实。六图传输完成不等于精度故障已修复。手机上次保存的运行 profile 为全 CPU；用户需重新打开 App、仍选择当前模型、切换离线 NPU 模式并重新加载后，执行上面的图文相关性测试。目前未自动运行真机图文测试。

用户随后要求改为从 ModelScope 重新下载。App 的 `modelManifestLoadableStatus` 会核对下载 manifest 中除 config 外的文件大小；此次旧图约 97 MiB、新图约 50 MiB，仅覆盖 OM 而未同步 App 下载记录会使加载状态不一致。已将完整候选发布为独立公开仓库 [fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_w8a8_output_int16](https://www.modelscope.cn/models/fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_w8a8_output_int16)。App 仓库输入填写该 `owner/repo`，正常下载完成后选择新模型，再切换离线 NPU 模式重新加载。

此次发布共 31 个文件、3,592,556,843 字节（约 3.35 GiB），包含 28 个完整运行文件、新 README、`offline_om_manifest.json` 及 `SHA256SUMS`。远端全部 31 个文件大小及 SHA-256 与本地一致，匿名文件清单及 config/六图的 resolve 下载地址均已验证；构建审计、量化中间数据、SDK、ENV.md 和凭证未上传。发布审计保存在 `/tmp/w8a8_modelscope_publish/upload_status.json`，固定范围发布脚本为 `/tmp/w8a8_modelscope_publish.py`。此轮发布未继续操作手机，也未执行图文精度测试；仓库明确标注为待验证候选。

## 安装 App 与传输测试包

1. 使用 DevEco Studio 打开 `/home/ma-user/workspace/feh/mobiinfra-oh`，构建并安装带调试签名的 App。此次只改 App 的 native 诊断和测试页，使用原有 `entry/libs/arm64-v8a/libMNN.so`。
2. 退出并重新打开 App，直接进入 **设置 → 调试工具 → Op Precision Test**，先不加载聊天模型或启动本地 Agent。若报告要求 unload local model，重新启动 App 后直接进入测试页即可。
3. 本地解压 ZIP。测试页的新路径输入框显示实际的 `context.filesDir + '/w8a8_probe'`。通过 DevEco Device File Explorer 将解压后的整个 `w8a8_probe` 文件夹上传到该 filesDir。路径框指向的目录内必须直接存在 `probe.tsv`、`inputs/`、`references/`、`om/`，不能再套一层同名目录。

也可以使用 HDC 的应用沙箱传输。以下目录对应此前日志中的 entry filesDir；如果测试页显示不同路径，以页面为准调整。`-b` 要求已启动、带调试签名的 App，并要求设备与 HDC 支持该选项，参见 [OpenHarmony 官方 HDC 文档](https://github.com/openharmony/docs/blob/master/zh-cn/application-dev/dfx/hdc.md#文件传输)。

```bash
hdc list targets
hdc shell -b com.coevomind.clawmate ls data/storage/el2/base/haps/entry/files
hdc file send -b com.coevomind.clawmate ./w8a8_probe data/storage/el2/base/haps/entry/files/
hdc shell -b com.coevomind.clawmate ls data/storage/el2/base/haps/entry/files/w8a8_probe
```

如果沙箱路径或 `-b` 不可用，用 Device File Explorer 上传到页面显示的路径。

## 真机执行与判断

点击 **Run W8A8 Fixed-input Probe**。它不受页面上 HiAI delegate / CPU precision 等选项影响：执行的是包中的离线图，采用与聊天相同的 `OfflineNpuChunkExecutor`，按模型描述处理输入顺序和 FP16/FP32。每次只保留一个实验图，每个样本都重新读取固定输入，不串接前一个 NPU 输出。

页面显示每个 case/sample 相对两个参考的 cosine、relative L2 和输出 absmax，结果保存到测试目录的 `results/`：

- `report.txt`：可复制的汇总。
- `report.tsv`：完整指标，包含两个参考的 absmax 和最大绝对误差。
- `<case>_<sample>.bin`：设备 hidden 输出，float32，可下载后逐元素比较。

`ALIGNED` 使用诊断阈值 cosine ≥ 0.999 且 relative L2 ≤ 0.02；它表示与该 CPU 参考接近，不能替代整个模型的图文精度验收。`MISMATCH` 表示需要分析数值差异，`ERROR` 表示读文件、加载、执行、输出长度或有限值检查失败。仅 `RunSync success` 不计作数值通过。

| 观察结果 | 下一步判断 |
|---|---|
| `float_control` 对浮点参考也明显不一致 | 先检查测试包、shape、设备兼容性和输入/输出描述，不归因于 A8 |
| 控制图对齐；baseline 对 W8A8 参考对齐，却对浮点参考偏离 | 支持 A8 本身造成误差，后续检查激活量化策略、校准和敏感层 |
| 控制图对齐；baseline 对两个参考均明显偏离 | 优先检查 compress_conf 被 OMG 消费的方式、量化轴、零点和 bias lowering |
| 仅 `no_bias_param` 明显改善 | 支持 MatMul 输入数量与 bias 参数匹配问题；仍需扩大样本和完整模型验证 |
| 仅 `pch` 明显改善 | 支持 per-channel 属性解释差异；仍需逐层确认权重量化轴 |
| 仅 `gemm` 明显改善 | 支持 MatMul+Add 与三输入量化结构的匹配问题 |
| 某种配置所有样本都改善 | 再验证 chunk1–5 和完整图文回答，不能只凭 chunk0 发布 |

下载结果示例（手机端路径以测试页为准）：

```bash
hdc file recv -b com.coevomind.clawmate data/storage/el2/base/haps/entry/files/w8a8_probe/results ./w8a8_results
```

回传 `results/`、包内 `manifest.json` 及 `OFFLINE_NPU` 的 `tensor_desc`、`input_values`、`output_values`、`run_sync_complete` 日志，即可进一步定位。聊天也新增了每个 chunk 的 `output_values role=hidden`：同一次请求中 chunk0 输出的元素数、min/max 和 hash 应与 chunk1 的 hidden 输入一致，这能检查交接是否引入变化。

## 历史实验测试

```bash
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
/home/ma-user/.conda/envs/CANN/bin/python -B -m unittest -v test_w8a8_probe.py
```

本机已通过 HDC 连接手机，并完成上述四轮固定输入测试；当前使用用户已安装的诊断 App。新增脚本的 7 个本机测试通过，覆盖权重布局、零点表示、显式补偿的浮点/量化数学、权重尺度同步调整及误差比较器。现已按用户要求停止继续试参；没有采用失败候选替换正式模型。bug 尚未修复；chunk0 对齐不能替代六个 chunk 和完整图文问答验收。

## 保留的配置回归测试

在仓库根目录运行，工具链路径以本地 ENV.md 为准：

```bash
export PYTHONPATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3:$PWD/transformers/llm/export
export PYTHONDONTWRITEBYTECODE=1
/home/ma-user/.conda/envs/CANN/bin/python -B -m unittest discover \
  -s transformers/llm/export/plugin_quant_visual_matmul_route_v1 \
  -p test_quant_config.py -v
```

四个测试覆盖激活符号的官方序列化、旧 checkpoint 覆盖配置、W4A16 分组和报告读取真实校准状态。测试辅助函数已内置于测试文件，使用临时目录与作用域内的 serializer hook，不再依赖已移除的实验构建脚本。它们验证配置与报告行为，设备精度仍以真机结果为准。

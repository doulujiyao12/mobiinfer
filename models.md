# 本机模型目录

更新日期：2026-11-12。

本文件记录本机模型地址，不保存访问令牌或其他凭据。每次下载、生成、保存或迁移模型后，更新对应的模型地址及运行入口。下载模型、量化权重、导出图、中间模型和模型压缩包统一保存在 `/temp/models/`；项目工作目录不保存这些文件。

## Mobi：离线单图 NPU 与在线 NPU

当前交付模型目录：

```text
/temp/models/mobi_2B_hybrid_kirin9030_mnn_20261018/
```

| 文件 / 入口 | 用途 |
| --- | --- |
| `config.json` | 离线 NPU：ViT 单图、图与权重分离，LLM 在 CPU 执行 |
| `config_online.json` | 在线 NPU：MNN 视觉 chunks 在线编译，LLM 在 CPU 执行 |
| `llm.mnn`、`llm.mnn.weight` | 两种模式共享的 CPU 语言模型 |
| `embeddings_bf16.bin` | CPU 词嵌入权重 |
| `vision/om/vision_full.omc` | 离线 ViT 图 |
| `vision/weights/d9805f4d5c900de433870a31/SubGraph_0.weight` | 离线 ViT 外置权重 |
| `visual_pre.mnn`、`visual_blocks_npu_0..5.mnn`、`visual_post.mnn` 及配套权重 | 在线 NPU 使用的完整 MNN 视觉模型 |

其他保留版本：

| 目录 | 说明 |
| --- | --- |
| `/temp/models/mobi_2B_offline_npu2_kirin9030_online_v2/` | 之前已发布的双模式运行包；当前使用优先选择上面的合并后交付包 |
| `/temp/models/mnn_mobi_2B_w8a8_visual_npu_kirin9030_pathA/` | 早期在线 NPU 导出目录 |
| `/temp/models/mobi_2B_full_npu_kirin9030_w4a16_g64_kv16_fusedvision_v1/` | ViT + LLM 全图 NPU 实验包，入口为 `config.json`；与 CPU LLM 混合方案不同 |

## Qwen3-VL-2B：MNN 主模型与 EAGLE-3

当前运行包目录：

```text
/temp/models/Qwen3-VL-2B-Instruct-Eagle3-HiAI-MNN/
```

| 入口 | 视觉执行 | 语言执行 |
| --- | --- | --- |
| `config.json` | 在线 HiAI NPU | CPU 主模型 + EAGLE-3 |
| `config_ar.json` | 在线 HiAI NPU | CPU 自回归，不启用 draft |
| `config_cpu.json` | CPU | CPU 主模型 + EAGLE-3 |
| `config_cpu_ar.json` | CPU | CPU 自回归，不启用 draft |

2026-11-12 原目录增加 Kirin9030 单图离线视觉：App 选择“离线 NPU”加载根 `config.json` 时保留 CPU EAGLE；`config_offline.json` / `config_offline_ar.json` 为独立离线入口，`config_online.json` / `config_online_ar.json` 保留无 marker 的在线配置。离线 ViT 为 FP16，三路 DeepStack 保留；原在线 W8 视觉及语言/draft 权重未重新量化。

主模型权重为 `llm.mnn.weight`，draft 权重为 `eagle.mnn.weight`，其余图、投影、tokenizer 和配置须随包保留。四种入口共享同一套权重。

来源与中间产物：

| 目录 / 文件 | 内容 |
| --- | --- |
| `/temp/models/Qwen3-VL-2B-Instruct/` | Qwen 官方原始模型；浮点权重为 `model.safetensors`，4,255,140,312 字节 |
| `/temp/models/Qwen3-VL-2B-Instruct-Eagle3-MNN/` | 官方 MNN 主模型和已转换 draft 的原始下载快照 |
| `/temp/models/Qwen3-VL-2B-Instruct-Eagle3-Draft/` | 原始 EAGLE-3 draft checkpoint |
| `/temp/models/mobiinfer-work/qwen3-vl-eagle-20261110/visual/` | 视觉导出的 MNN / ONNX 图与权重等中间产物 |
| `/temp/models/archives/qwen3-vl-2b-eagle3-online-hiai-model.zip` | 已生成的 Qwen 运行模型压缩包 |

项目中的 `outputs/qwen3-vl-2b-eagle3-hiai/` 仅保留原生库、头文件、说明和验证记录；不再保留 `model/` 副本或模型 ZIP。旧 `.artifacts/` 中的模型目录已迁走，后续命令使用本文件的新地址。

## Qwen3.5-2B 与原生 MTP

| 目录 | 内容与状态 |
| --- | --- |
| `/temp/models/Qwen3.5-2B/` | Qwen 官方浮点 checkpoint，包含 `mtp.*` 权重；已下载并按官方 ModelScope 文件清单核对 SHA256，见 `download_manifest.json` |
| `/temp/models/mobiinfer-work/qwen35-mtp-20261111/` | 本次导出与验证工作目录；`reference/` 保存原始 HF CPU 图文参考，`mapping/` 保存主模型与 MTP 映射对照，`dequant_reference/` 保存实际导出权重的反量化对照报告 |
| `/temp/models/mobiinfer-work/qwen35-mtp-20261111/export/` | 主模型 W4、原生 MTP W4 和视觉模型的 MNN / ONNX 导出与视觉参考数据 |
| `/temp/models/mobiinfer-work/qwen35-mtp-20261111/vision_fp16/` | 独立从相同视觉 ONNX 转换的 FP16 候选，最终特征对 HF 的相对 L2 误差约 0.29% |
| `/temp/models/mobiinfer-work/qwen35-mtp-20261111/vision_w8/`、`vision_fp16_c4_probe/` | 前期视觉精度 / 布局对照候选，仅保留实验记录，不用于最终部署 |
| `/temp/models/mobiinfer-work/qwen35-mtp-20261111/export_final/` | 最终脚本的全流程导出目录，视觉 FP16、LLM/MTP W4；与前期 W8 视觉候选分开保存 |
| `/temp/models/mobiinfer-work/qwen35-mtp-20261111/validated_candidate/` | 前期已完成 CPU 验证的候选运行包；用于与最终脚本导出逐文件核对 |
| `/temp/models/Qwen3.5-2B-MTP-HiAI-MNN/` | 最终运行包：`config.json` 为在线 HiAI 视觉 + CPU 原生 MTP；`config_ar.json` 关闭 MTP；`config_cpu.json` / `config_cpu_ar.json` 为 CPU 对照入口。主机图文与状态验证已通过，尚未真机验证 |
| `/temp/models/mobiinfer-work/qwen35-mtp-20261111/publication/` | ModelScope 发布目录，以硬链接保留最终运行文件，另附中文模型说明、鸿蒙 ARM64 库和验证报告；不修改已验收的权重 |

新模型采用 HQQ 权重量化：LLM / MTP 主干 W4/group64，共享词嵌入与 LM head W8/group64；视觉保留 FP16。CPU 采用 `dynamic_option=2` 分组在线激活量化，默认 `precision=high` 使递归状态与普通 KV 保持 FP32。未进行新的 GPTQ / 图片激活校准；少量问答通过不代表与原始模型无损对齐。

最初在线包已发布到 [fengerhu1/Qwen3.5-2B-MTP-HiAI-MNN](https://modelscope.cn/models/fengerhu1/Qwen3.5-2B-MTP-HiAI-MNN)，历史 37 文件发布记录保存在 `outputs/qwen3.5-2b-mtp-hiai/publication/`。2026-11-12 同仓库更新为在线 / 离线 / CPU 完整包，新增 FP16 单图离线视觉；原 CPU LLM/MTP 和在线视觉保持不变。原目录也同步更新，App 离线模式可直接使用根配置并保留 MTP，另有 `config_offline*.json` 和 `config_online*.json`。

本轮两个仓库的远端大小与 SHA256 均已逐文件核对：Qwen3-VL 共 **54 文件 / 2,584,839,827 字节**，Qwen3.5 共 **49 文件 / 3,233,991,041 字节**。发布记录分别在 `outputs/qwen-offline-npu-20261112/publication/qwen3vl/` 和 `qwen35/`，本机同步结果为对应 `local_promotion.json`。远端 `runtime/` 提供新的鸿蒙库、头文件和离线桥接源码 ZIP，必须配套重编 App；没有模型权重混入该 ZIP。尚未真机验收。

## 后续下载与导出

本轮单图离线 ViT 构建：`/temp/models/mobiinfer-work/qwen_offline_20261112/qwen3vl/` 与 `/temp/models/mobiinfer-work/qwen_offline_20261112/qwen35/` 分别存放两份模型的 FP16 视觉 ONNX、OMC、外置权重和 CPU 对照数据。编译器临时图和权重统一放在 `/temp/models/compiler_work/`。新双模式运行文件副本位于 `/temp/models/mobiinfer-work/qwen_offline_20261112/packages/<模型名>/`；与远端一致的完整发布目录位于同一工作目录的 `publication/<模型名>/`，旧版元数据备份位于 `baselines/<模型名>/`。两个 `<模型名>` 分别是 `Qwen3-VL-2B-Instruct-Eagle3-HiAI-MNN` 和 `Qwen3.5-2B-MTP-HiAI-MNN`。正式使用仍选上面 `/temp/models/` 下的原模型目录；不可变大文件通过硬链接复用，JSON / 说明 / 清单独立保存。

同一工作目录的 `repack_validation/<模型名>/` 用于验证最新双模式包可再次安全打包，保持在线配置和已验收资产；不是新的精度候选。

- Hugging Face 下载缓存放在 `/temp/models/hf_home/`，ModelScope 下载缓存放在 `/temp/models/modelscope_cache/`；下载时显式设置本地模型目录。
- 临时导出、量化、校准模型放在 `/temp/models/mobiinfer-work/`，模型压缩包放在 `/temp/models/archives/`。
- 二进制权重相同的副本可通过硬链接复用空间；不要原地覆写共享的权重文件。需要新版本时使用新目录和新文件。
- 本机目录约定不改变手机 App 沙箱中的模型路径，也不改变已发布的 ModelScope 模型。

# DDK 6.1.1.0 环境补齐说明（W4A16 → omc 编译环境修复）

> 本文档记录对 `/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete/` 补全目录所做的全部修改，
> 以及修改的原因。**未改动新旧两套 DDK 的原始目录，也未改动任何代码/脚本文件。**

- 日期：2026-09-06
- 操作人：Claude（会话名 `huanjing`）
- 目标：让新 DDK 6.1.1.0 具备完整的 AscendC 算子编译环境，用于 W4A16 量化模型的 omc 编译。

---

## 1. 背景与问题

### 1.1 需求链路

用户希望用 **W4A16**（`Quant_act_weight_eco`，权重 4bit / group 128，激活 16bit）量化 6 个 visual
chunk，并用 `--target=omc` 输出 `.omc` 格式（平台 `kirin9030`）。

涉及的两套 DDK：

| 名称 | 路径 | 用途 |
|------|------|------|
| 新 DDK 6.1.1.0 | `/temp/fdh/ddk/DDK-tools-next-6.1.1.0/` | 用户要求升级后使用（dopt 量化工具） |
| 旧 DDK 6.0.1.0 | `/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/` | 原有完整环境 |

### 1.2 问题定位

用新 DDK 6.1.1.0 的 `omg` 做 `--target=omc` 编译时，报错：

```
ModuleNotFoundError: No module named 'te_fusion'
PlugIn library :libai_npucore_generated.so can't dlopen ...
E/AI_NPUCL: initialize lib adaptee: ascendc_lib failed
```

逐步排查后确认，新 DDK 6.1.1.0 的包结构与旧 DDK 不同，**缺少 omc 编译 AscendC 算子所必需的组件**。

---

## 2. 新/旧 DDK 结构差异

### 2.1 顶层目录差异

| 目录 | 旧 DDK 6.0.1.0 | 新 DDK 6.1.1.0 |
|------|----------------|----------------|
| `tools/` | ✅ | ✅ |
| `ddk/`（含 `ccec_compiler`、`ai_ddk_lib`、`tikcpp`） | ✅ | ❌ **缺失** |

### 2.2 AscendC python 模块差异

omc 编译 AscendC 算子需要 `te_fusion` / `te` / `tbe` 这些 python 模块。

- 旧 DDK：`tools/tools_ascendc/package/python/` 已解压好（含 `te_fusion`、`te`、`tbe`）
- 新 DDK：`tools/tools_ascendc/package/` 里只有 `ascendc_adapter-0.1-py3-none-any.whl`（未解压）

### 2.3 omg wrapper 脚本的路径引用差异

`tools/tools_omg/omg`（bash 包装脚本）会引用以下 `ddk/` 子路径：

| 引用 | 旧 6.0.1.0 | 新 6.1.1.0 |
|------|-----------|-----------|
| `ddk/tbe/lib64/` | ✅ 有 | ✅ 有（但 ddk/ 目录缺） |
| `ddk/ascendcpp/lib64/` | ❌ 无 | ✅ **新增** |
| `ddk/ascendcpp/conf/`（`KIRIN_CONF_JSON_PATH`） | ❌ 无 | ✅ **新增** |
| `ddk/ccec_compiler/bin/` | ✅ 有 | ✅ 有（但 ddk/ 目录缺） |

> 结论：新 DDK 6.1.1.0 的 omg 新增了对 `ddk/ascendcpp/` 的依赖（新架构），
> 但它的发行包里没有 `ddk/` 目录，AscendC 算子库实际放在 `tools/platform/kirin9030/` 下。

---

## 3. 所做的修改

为避免污染新旧两套 DDK 的原始目录，**创建了一个独立的补全目录**：

```
/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete/
```

### 3.1 复制新 DDK

```bash
cp -a /temp/fdh/ddk/DDK-tools-next-6.1.1.0 \
      /temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete
```

### 3.2 建立 `ddk/` 子目录软链（指向补全目录自身内部）

`omg` wrapper 需要 `ddk/ascendcpp`、`ddk/ccec_compiler`、`ddk/tikcpp`，而对应内容其实已存在于补全
目录的 `tools/` 内部。因此建立软链（**相对路径，指向补全目录自身，不依赖旧 DDK**）：

| 软链 | 目标（补全目录内部） |
|------|----------------------|
| `ddk/ascendcpp` | `../tools/platform/kirin9030` |
| `ddk/ccec_compiler` | `../tools/tools_ascendc/bisheng` |
| `ddk/tikcpp` | `../tools/tools_ascendc/include/tikcpp` |

其中 `ddk/ai_ddk_lib` 为真实目录（复制自旧 DDK 的 `ddk/ai_ddk_lib`，内容很小）。

### 3.3 解压新 DDK 自带的 AscendC python 模块

用**新 DDK 自己的** `ascendc_adapter` whl（而非旧 DDK 的），解压到 `package/python/`：

```bash
mkdir -p tools/tools_ascendc/package/python
cd tools/tools_ascendc/package/python
unzip -o ../ascendc_adapter-0.1-py3-none-any.whl
```

解压得到 `te_fusion/`、`te/`、`tbe/`、`ascendc_adapter-0.1.dist-info/`。

> 为什么用新 DDK 的 whl 而不是复制旧 DDK 的 `package/python`：
> 旧 DDK 的 `package/python` 是 6.0.1.0 版本，可能与 6.1.1.0 的 omg 二进制存在版本不匹配风险。
> 新 DDK 自带 `ascendc_adapter-0.1-py3-none-any.whl` 才是与 6.1.1.0 配套的正确版本。

---

## 4. 补全后的目录结构

```
/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete/
├── ddk/
│   ├── ai_ddk_lib/               # 真实目录（来自旧 DDK）
│   ├── ascendcpp -> ../tools/platform/kirin9030          # 软链
│   ├── ccec_compiler -> ../tools/tools_ascendc/bisheng   # 软链
│   └── tikcpp -> ../tools/tools_ascendc/include/tikcpp   # 软链
└── tools/
    ├── platform/kirin9030/       # AscendC 算子库 + config（原新 DDK 自带）
    ├── tools_ascendc/
    │   ├── bisheng/              # ccec 编译器（原新 DDK 自带）
    │   ├── include/tikcpp/       # tikcpp（原新 DDK 自带）
    │   └── package/python/       # ★ 新增：te_fusion/te/tbe（从 whl 解压）
    ├── tools_dopt/               # dopt 量化工具（原新 DDK 自带）
    └── tools_omg/                # omg 编译器（原新 DDK 自带）
```

---

## 5. 使用方式（不改脚本，通过环境变量）

原脚本 `run_visual_plugin_matmul_omc_omc.sh` 支持通过环境变量 `OMG_TOOL`、`OMG_MASTER_DIR` 覆盖
工具路径。因此**无需修改任何脚本**，在运行前 export 以下环境变量即可指向补全目录：

```bash
export DDK_PATH=/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete
export TOOLCHAIN_HOME=$DDK_PATH
export PYTHONPATH=$DDK_PATH/tools/tools_ascendc/package/python:$PYTHONPATH
export LD_LIBRARY_PATH=$DDK_PATH/tools/tools_omg/master/lib64:$DDK_PATH/tools/platform/kirin9030/lib64:$LD_LIBRARY_PATH
export PATH=$DDK_PATH/tools/tools_omg/master:$DDK_PATH/tools/tools_ascendc/package:$PATH
export OMG_TOOL=$DDK_PATH/tools/tools_omg/omg
export OMG_MASTER_DIR=$DDK_PATH/tools/tools_omg/master
```

（量化工具 `DDK_DOPT` 仍指向 `/temp/fdh/ddk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3`，
也可改用补全目录下同样的路径。）

---

## 6. 验证结果

### 6.1 环境修复成功 ✅

用补全目录编译最小 `Gemm` 模型（`--target=omc`，`--platform=kirin9030`）：

```
I/ASC: Get Kernel Bin name for node [fc_gemm, type:GemmV3] : GemmV3_9953b0d0...
I/AI_FMK: SaveCompiledModelToFile SUCCESS.
I/OMG_TOOL: OMG generate offline model success.
```

产出了 **110KB 的 `.omc`**，日志显示 `GemmV3` 算子**生成了真正的 NPU kernel**。
证明补全后的 DDK 环境已能正常初始化 AscendC 算子库并编译算子。

### 6.2 W4A16 量化 omc 仍失败 ❌（平台算子限制，非环境问题）

对真实 W4A16 量化图（`Gemm` + `--compress_conf`）编译时：

```
CheckSupported: op [blocks.0.mlp.linear_fc1] type [GemmD] is not supported in npucl store [fe_lib]
Node blocks.0.mlp.linear_fc1 type GemmD don't support!
check ir model compatibility failed
```

原因：带量化参数时，omc 编译器把 `Gemm` 映射为量化算子 `GemmD`，而 **`GemmD` 在 kirin9030 的
`fe_lib` 算子库里不被支持**（`npu_ascendc_opinfo.json` 中没有 `GemmD`）。

> 这是 **kirin9030 平台算子库的硬限制**，与 DDK 环境无关，也无法通过换算子（MatMul/Gemm/其他）解决。

---

## 7. 关键结论

1. **环境问题已修复**：补全目录 `/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete/` 能正确编译
   AscendC 算子（`GemmV3` 生成 NPU kernel 成功）。

2. **W4A16 + omc 仍不可行**：kirin9030 的 omc 编译器缺少量化算子 `GemmD`，W4A16 量化模型无法
   编译为 omc。这是平台算子库限制，不是脚本或环境问题。

3. **未量化 fp16 + omc 可行**：fp16 未量化的 `Gemm` 图可以成功编译为 omc（实测通过）。

---

## 8. 未改动清单（确认无污染）

- ❌ 未修改 `/temp/fdh/ddk/DDK-tools-next-6.1.1.0/`（新 DDK 原目录）
- ❌ 未修改 `/temp/fdh/baiducloud/.../DDK-tools-next-6.0.1.0/`（旧 DDK 原目录）
- ❌ 未修改任何脚本文件（`run_real_calib_16_W4A16_omc.sh`、`run_visual_plugin_matmul_omc_omc.sh` 等）
- ✅ 仅新增独立补全目录 `/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete/`

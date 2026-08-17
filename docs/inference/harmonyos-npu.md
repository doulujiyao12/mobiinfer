# MNN 在鸿蒙（HarmonyOS）NPU 上的运行机制详解

> 本文基于当前仓库（`main` 分支）的实际代码，详细说明 MNN 如何调用华为 HiAI NPU（昇腾 Da Vinci 架构），
> 从 MNN 图到 NPU 可执行图（OM）的转换过程，以及 CPU 与 NPU 的协作方式。
> 相关代码全部位于 `source/backend/hiai/` 与 `transformers/llm/engine/`。

---

## 1. 总览：仓库中有四条 NPU 使用途径

MNN 是一个"图优化 + 异构后端调度"的推理引擎。在鸿蒙设备上，NPU 通过华为 **HiAI DDK**
（`libhiai.so` / `libhiai_ir.so` / `libhiai_ir_build.so`，即 HiAI Model Manager / IR Builder）暴露能力。
本仓库提供了 **四条** 使用 NPU 的途径，按注册的 `MNNForwardType` 区分：

| 途径 | 名称（配置串） | ForwardType | 位置 | 说明 |
|---|---|---|---|---|
| ① 全图 HiAI 后端（在线编译） | `hiai` | `MNN_FORWARD_USER_0` | `source/backend/hiai/backend/NPUBackend.cpp` | 整张子图一次性翻译成 ge::Graph，**设备端在线编译**为 OM 后在 NPU 上执行；不支持的 op 回退 CPU |
| ①′ ① + OM 缓存 | `MNN_HIAI_CACHE_OM_BY_CHUNK` | 同 ① | 同 ①（`NPUBackend.cpp:1089-1155`） | 在线编译成功后把 OM 落盘，下次启动直接加载，跳过 BuildIRModel；shape-aware 签名自动失效 |
| ② HiAI 逐算子委托 | `hiai_delegate` | `MNN_FORWARD_USER_1` | `source/backend/hiai/backend/HiAIDelegateBackend.cpp` | 会话主体仍是 CPU，仅把 Convolution（含量化变体）逐个委托给 NPU，每个 conv 单独编译一个 HiAI 小模型 |
| ③ OM 直通执行器 | 应用层插件 | （不在 MNN 后端列表） | `transformers/llm/engine/include/llm/npu_chunk_executor.hpp` | **预编译好的 `.om` 文件**由 HarmonyOS 应用层（`HIAIModelManager`）直接执行，MNN 只负责编排与数据桥接 |

> 另外注意：**OMG 不是第四条运行时途径**，而是"离线生成 `.om`"的 PC 工具流程
> （`HIAI_DEBUG` 导出 `test.irpb` + DDK/CANN command-line-tools 离线编译），产物供途径 ③ 使用，
> 见 §1.2 与 §7。

路径 ③ 是纯接口（`INpuChunkExecutor`），实现放在 HarmonyOS 应用层，MNN 通过虚函数表调用，
因此 MNN 引擎代码不需要任何鸿蒙头文件：

```cpp
// transformers/llm/engine/include/llm/npu_chunk_executor.hpp
class INpuChunkExecutor {
public:
    virtual bool loadChunk(int chunkIdx, const std::string& omPath) = 0;
    virtual bool runChunk(int chunkIdx,
                          const std::vector<float>& hiddenInput,  // [S * hiddenDim]
                          const std::vector<float>& rotaryInput,  // [2 * S * headDim]
                          const std::vector<float>& maskInput,    // [S * S]
                          std::vector<std::vector<float>>& outputs) = 0;
    virtual void unload() = 0;
};
```

> 头文件注释明确写道："The HarmonyOS app layer implements this with HIAIModelManager (OM)."

### 当前仓库落地场景：哪些部分用到了 NPU

以仓库配套的多模态 LLM（视觉 + 语言，`omni.cpp` / `llm.cpp`）为例：

- **视觉编码器（ViT）blocks**：跑在 NPU 上（路径① 或 ③，取决于配置）；
- **视觉 pre/post 处理**：跑在 CPU（`mProcessorRuntimeManager`）；
- **语言模型 decoder**：通常跑在 CPU；若配置 `backend_type=hiai` 也可整模型走路径①，不支持的 op 由 CPU 兜底；
- **hiai_delegate 模式**：decoder 或任意模型里"长得像 Linear 的 1×1 卷积"可被逐算子委托到 NPU 的
  Da Vinci CUBE 矩阵单元（路径②，这是纯 perf A/B 手段，默认有 70 个 op 上限）。

### 1.1 四条途径的机制与优劣对比

**途径 ①（在线编译）**——当前主路径，本仓库模型的默认走法：

```
MNN 图 → 每 op 的 Execution 搭 ge 子图（mGrapMap）→ bulidIRModelAndLoad()
  → CreateModelBuff + BuildIRModel（设备端 DOMI 运行时编译 IR→OM）→ LoadModelSync → Process
```

- **优**：全自动，MNN 图直接可用，无外部工具链；shape 在每次 onResize 重新编译，动态灵活；
- **劣**：冷启动编图 30s–5min（大图）；编图期峰值内存高——`model.Save` 序列化可达 3GB+，
  会被鸿蒙 watchdog OOM kill，所以平时跳过 Save（`NPUBackend.cpp:1175` 仅 HIAI_DEBUG 下执行）。

**途径 ①′（在线编译 + OM 缓存）**——① 的生产化补丁（`MNN_HIAI_CACHE_OM_BY_CHUNK`，build_64.sh 已开）：
在线编译成功后 `WriteToOMFile` 存到 `<npu_model_dir>/chunk_i/om_cache_v2/<shape_key>/vision.om`
（`NPUBackend.cpp:1209`）；下次启动 fast path 直接读文件 `LoadModelSync`，**跳过 BuildIRModel**
（`NPUBackend.cpp:1120-1146`）。

- **优**：保留①的全自动 + 跨启动免编译（后续启动秒级）；缓存 key 是 shape-aware 的——
  签名含 op 数量、chunk 名、输入/输出 shape/type/layout（`NPUBackend.cpp:846-866`），
  shape 变化自动 miss 重编；文件不可读/加载校验失败自动删缓存重编（`NPUBackend.cpp:1133-1152`）；
- **劣**：首次仍需在线编译；签名**不含权重内容**——模型权重更新后不会自动失效，
  需手动删除缓存目录（或换 chunk 名），否则会加载旧权重的 OM（结果错误而非精度下降）。

**途径 ②（per-op 委托）**：`HiAIDelegateBackend` 继承 `CPUBackend`，onCreate 拦截
`OpType_Convolution` → `HiAIConvExecution` 每个 conv **单独搭 ge 子图 + 单独 BuildIRModel +
单独 LoadModelSync**（一个 conv 一个小 OM，上限 70 个）。

- **优**：op 级回退（不支持的 conv 自动走 CPU），无需整图支持；
- **劣**：每 conv 一个模型加载、无算子融合、上限 70 个，性能远逊全图，大模型不适用。

**途径 ③（应用层离线 OM 直通）**：app 层实现 `INpuChunkExecutor`（`HIAIModelManager` 的 OM
接口），`omni.setNpuChunkExecutor(executor, omPaths)` 注入（`omni.hpp:183`）；omni 加载时对
`omPaths[i]` 非空的 chunk 调 `loadChunk(i, xx.om)`，**MNN 完全绕过 NPUBackend/在线编译**
（`omni.cpp:412-418`）；执行走 `runChunk`，omni.cpp 用 `varpToFloatVector/floatVectorToVarp`
在 VARP 与 float 数组之间桥接（`omni.cpp:915-945`）。支持 sparse 混合（omPaths[i] 为空则该
chunk 走 MNN 模块），deepstack 合并输出用配置键 `visual_blocks_om_deepstack_dup`。

- **优**：零在线编译（启动秒级）、峰值内存最低、.om 可预置随包分发；
- **劣**：shape 固定（离线编什么 shape 就只能跑什么 shape，输入变化要重编）；需要 app 层完整
  实现 executor；MNN 侧无编译期校验。

|  | ①在线编译 | ①′在线+缓存 | ③离线OM直通 | ②per-op委托 |
|---|---|---|---|---|
| 编译位置 | 设备端 | 设备端（仅首次） | **PC 离线** | 设备端（每 conv） |
| 冷启动 | 30s–5min | 首次慢，之后秒级 | 秒级 | 每次首帧慢 |
| 峰值内存 | 高 | 首次高，之后低 | 最低 | 低 |
| shape 灵活性 | 动态 | 动态（自动 miss 重编） | **固定** | 动态 |
| 权重更新 | 无影响 | 需手动清缓存 | 需重新离线编 | 无影响 |
| 外部依赖 | 无 | 无 | OMG 工具链 + app 层 executor | 无 |
| 适用 | 开发迭代 | **生产端侧（推荐）** | 量产首发/启动敏感场景 | 小模型加速 |

### 1.2 离线 OMG：`.om` 的生成工具（非运行时途径）

"OMG"（Offline Model Generator）是昇腾/CANN 侧的离线模型转换工具族（`hiai_ir_build.h`
头文件注释 `@ingroup domi_omg` 即指此）。它**不在 MNN 运行时代码里**，而是途径 ③ 的 `.om`
来源，流程为：

1. 用 `HIAI_DEBUG` 构建（或临时打开 HIAI_DEBUG）在设备上跑一次模型，
   `model.Save(buffer)` 把 ge 图序列化成 `test.irpb` 落盘（`NPUBackend.cpp:1175`；
   平时跳过 Save 是为省 3GB+ 序列化内存）；
2. 把 irpb 拷到 PC，用 DDK/CANN 的 command-line-tools（见 `ENV.md`：
   `/temp/csm/command-line-tools/`）离线编译成 `.om`；
3. `.om` 随 app 打包，运行时走途径 ③ 由 `INpuChunkExecutor` 直接加载执行。

### 1.3 offline_hiai_npu 分支特有功能：PC 端 OMG 离线编译闭环

`offline_hiai_npu`（远端分支，HEAD 为 `ab039afc`，与 `feh_dev` 分叉于 `798dbf4d`）额外实现了
一条 **PC 端全离线圈住**的图编译管线：视觉块在 PC 上用昇腾 OMG 工具直接编成 `.om`，设备端
零 IR build。独有内容：

1. **`transformers/llm/export/plugin_quant_visual_matmul_route_v1/`**（整个目录为该分支独有）：
   - `visual_plugin_quant_matmul_route.py`（1161 行）：把视觉 blocks 改造成可量化的 route ONNX，
     并生成 W8A8 量化参数；
   - `run_visual_plugin_matmul_omc.sh`：直接调用昇腾 **OMG 工具**（`DDK-tools-next-6.0.1.0/tools/tools_omg/omg`）：
     `--model <route.onnx> --framework 5 --target=om --input_shape=... --weight_data_type=fp16|fp32
     --compress_conf <quant_params> --platform=kirinx90` —— **编译 + 量化都在 PC 上完成**；
   - `rename_om_files.sh`：把 OMG 产物 `visual_plugin_matmul_quantized.om` 重命名为
     `visual_blocks_npu_<i>.om`（对齐 config 的 chunk 命名，约 49.9 MB/个）；
   - 校准闭环：`MNN_VISUAL_CHUNK_INPUT_DUMP`（omni.cpp/omni.hpp 编译期选项，dump 真实 chunk
     输入 hidden/rotary/mask 到磁盘）+ `collect_visual_act_stats_v6.py`（act stats 收集）+
     `eval_chunk_quant.py`（cos/MSE/absmax 评估报告）+ `run_all_chunks_real_calib*.sh`；
   - **pillow resize 位级对齐**：`smartResize` + `pillowBicubicFilter`（round-half-even），
     保证 dump 的校准输入与 Python 端预处理 bit-exact；
   - 意义：编译与量化全流程在 PC 完成，设备端零在线编译（省首帧 30s–5min 与 3GB+ 编图内存）。
2. **`source/backend/hiai/custom/LayerNormCustomOp.hpp`**：用 `HIAI_REG_OP` 注册自定义 HiAI 算子
   `LayerNormCustom`（输入 x/gamma/beta，属性 epsilon/norm_size），`NPULayerNorm.cpp` 改用它
   （feh_dev 侧是 `#if MNN_HIAI_LN_USE_CUSTOM` 保护）——绕开 NPU 原生算子对 LayerNorm 的限制；
3. **CPU int4/int8 MatMul ARM 汇编**（`MNNPackedMatMul_int4/int8.S` 等 8 个）——配套量化模型
   在 CPU 侧执行；
4. 该分支**没有** `feh_dev` 的 chunk 缓存（①′）、加载名/执行名修复与 dflash。

两分支关系：运行时加载 `.om` 的机制完全一致（都是 `INpuChunkExecutor`）；差异只在"`.om`
从哪来"——feh_dev 靠设备端在线编译（+①′缓存），offline 分支靠 PC 端 OMG 直编（含量化）。

### 1.4 三个常见疑问的代码级澄清

#### 1.4.1 NPU 输出顺序是否固定？hidden 与 deepstack 同维度怎么区分？

**结论：输出顺序不是由 shape 决定的，同维度输出只能靠"约定顺序 + 实际验证/配置"**，代码
里三处印证：

- `NPUBackend.cpp:1006-1009`（getInOutTensorInfo）：明确注释 "We cannot assume 1:1 index
  correspondence because the DDK may produce a different number of outputs (e.g. when
  hidden_states and deepstack originate from the same DDK node, the DDK may merge them into
  one output buffer)"，随后按**字节大小匹配**：先给未占用的 DDK slot 按大小分配，再按出现
  顺序 zip——同字节大小的输出（hidden 与 deepstack 维度相同 → 字节相同）只能按顺序对应；
- `omni.cpp:955-966`（路径③）：约定 `omOutputs[0]=hidden`、`omOutputs[1:]=deepstack`；当
  OM 编译把 hidden==deepstack **合并成同一个输出**时，靠配置键 `visual_blocks_om_deepstack_dup`
  列出需要把 output[0] 复制一份当 deepstack 的 chunk 索引（`omni.cpp:424-427`）——这个
  配置就是"手动试出来"的结果；
- `eval_chunk_quant.py:146-151`（offline 分支）：约定 "第一个是 hidden，其余是 deepstack_k"。

**实践**：路径③（app 层 OM 直通）最明显——OM 的输入/输出顺序由 OMG 编译决定，必须实际
跑一遍（或对照 fp 基线输出逐张量算 cos）验证；路径①（MNN 侧）由 MNN 按字节大小+顺序
匹配，若顺序错乱会静默算错，同样需要端到端验证。

#### 1.4.2 哪些路径用"OH 系统 API"加载 OM，哪些用 `mnn::hiai`（HiAI DDK）？

**本仓库（MNN）内没有任何 OH_*/ohos 头文件代码**（已 grep 验证）。加载 OM 的两类 API 家族：

| 家族 | 组件 | 用在哪条路径 |
|---|---|---|
| **MNN 内嵌 HiAI DDK**（进程内直接链接 `libhiai.so` / `libhiai_ir.so` / `libhiai_ir_build.so`） | `hiai::AiModelMngerClient`（`HiAiModelManagerService.h`，namespace hiai）+ `domi::HiaiIrBuild` + `ge::graph` | 路径①（整 chunk 图 BuildIRModel→LoadModelSync）、①′（缓存 OM 文件再 LoadModelSync）、②（每 conv 一个小模型）——**全部走 mnn::hiai graph** |
| **鸿蒙系统 HIAI 服务 API**（`HIAIModelManager`，OH 体系接口） | 由 HarmonyOS 应用层实现 | 路径③的 `INpuChunkExecutor` 实现——`npu_chunk_executor.hpp` 注释明确："The HarmonyOS app layer implements this with HIAIModelManager (OM)"。本仓库只有虚函数接口，实现在 app 仓库（mobiinfer-oh） |

即：**加载"自己编译/缓存的 om"走 mnn::hiai DDK（路径①/①′/②）；加载"外部预编译的 om
（OMG 产物）"走 app 层的 OH 系统 API（路径③）**。OMC 不是格式，见下。

#### 1.4.3 OMG 是干什么的？和 om / omc 的区别？

- **OMG（Offline Model Generator）**：昇腾/CANN 侧的 **PC 端离线模型转换工具**（DDK-tools
  的 `tools_omg/omg`，`hiai_ir_build.h` 注释 `@ingroup domi_omg`）。输入 ONNX/Caffe 模型
  （`--framework 5` = ONNX），输出昇腾 NPU 可执行模型；支持 `--compress_conf`（量化参数）、
  `--weight_data_type`、`--platform`（指定 SoC，如 kirinx90）。本仓库在 `run_visual_plugin_matmul_omc.sh`
  中直接调用它完成"离线编译 + 量化"。
- **OM（Offline Model）**：昇腾 NPU 的**可执行模型文件**——设备端在线 `BuildIRModel` 的
  产物与 PC 端 OMG 的产物**同构**（都是 OM）；MNN 运行时加载、缓存的都是 OM。
- **OMC 不是独立文件格式**：它是 offline 分支工作流里对"OMG 编译产物"的**目录/命名习惯**
  （`omc_output/` = OM Compiled 输出目录，脚本打印 `Done: ...omc`），实际落盘文件名是
  `visual_plugin_matmul_quantized.om`（`rename_om_files.sh` 处理的正是 `.om` 文件），运行时
  路径③加载的也是 `.om`。

一句话：**OMG 是工具（PC 端把模型编成 OM），OM 是产物（NPU 可执行模型），omc 只是这套
工作流对 OMG 产物目录的习惯叫法**。

---

## 2. 代码布局

```
source/backend/hiai/
├── CMakeLists.txt                    # MNN_NPU 编译单元，链接 3rdParty 的 HiAI DDK 库
├── .gitignore                        # 3rdParty 库文件不入库（本地放置）
├── backend/
│   ├── NPUBackend.hpp / .cpp         # 路径①：NPU Runtime + Backend（核心，~1300 行）
│   └── HiAIDelegateBackend.hpp/.cpp  # 路径②：CPU 子类 + per-op 委托
└── execution/                        # 52 个 op 的 NPU 实现（NPUXxxExecution 模式）
    ├── NPUCommonExecution.hpp/.cpp   # 所有 NPU op 的基类（持有 NPUBackend*）
    ├── HiAIConvExecution.hpp/.cpp    # 路径② 的卷积委托执行器（~1300 行）
    ├── NPUConvolution.cpp            # 卷积（含 Linear→MatMul 特判）
    ├── NPUAttention.cpp              # LLM Attention（QK^T→Mask→Softmax→×V）
    ├── NPUMatmul.cpp / NPUBatchMatMul.cpp / NPULayerNorm.cpp ...
    └── NPUxxx.cpp（共 50 余个文件）
```

第三方依赖（不入库，需按 ENV.md 放置）：`source/backend/hiai/3rdParty/<ABI>/libhiai.so`、
`libhiai_ir.so`、`libhiai_ir_build.so` 以及 `include/` 下的 DDK 头文件
（`HiAiModelManagerService.h`、`hiai_ir_build.h`、`graph/op/all_ops.h` 等）。

构建开关（根 `CMakeLists.txt`）：

```cmake
IF(MNN_NPU)
    if (CMAKE_SYSTEM_NAME MATCHES "^Android") set(HIAI_PATH ${ANDROID_ABI})
    if (OHOS)                           set(HIAI_PATH ${OHOS_ARCH})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/)
    # 链接 hiai / hiai_ir / hiai_ir_build 三个 DDK 库
ENDIF()
# 另有：
option(MNN_HIAI_CACHE_OM_BY_CHUNK "Enable chunk-aware OM cache path for HiAI backend" OFF)
option(HIAI_VERBOSE "Enable HiAI verbose log" OFF)
```

---

## 3. 路径①：全图 HiAI 后端（`MNN_FORWARD_USER_0` / `hiai`）

### 3.1 Runtime 注册与启动自检

`NPUBackend.cpp:1278` 用全局初始化器把 `NPUBackendCreator` 注册到
`MNNInsertExtraRuntimeCreator(MNN_FORWARD_USER_0, ...)`。创建 Runtime 时做自检：

1. 创建 `hiai::AiModelMngerClient` 并 `Init(nullptr)`，失败则返回 nullptr（后端不可用）；
2. 读取 DDK `GetVersion()`，版本 ≤ `"100.330.000.000"` 视为不支持，返回 nullptr。

`NPURuntime::onGetCompilerType()` 返回 `Compiler_Origin` —— 表示**不做几何分解**，
NPU 后端消费的是 MNN 原始算子（Origin Op），每个算子一个 `Execution`。

`NPUBackend` 本身是 `MNN_FORWARD_USER_0` 类型的 `Backend`；`NPURuntime::onCreate()` 创建它。

### 3.2 op 注册表：52 个支持类型

每个 NPU op 实现以 `NPUCreatorRegister` / `TypedCreator` 模板注册到 `NPUBackend` 的静态 Creator 表
（`getCreatorMap()`），映射 `OpType → Execution 构造器`。支持类型（全表见 `execution/` 下的静态注册）：

- **卷积族**：Convolution / ConvolutionDepthwise / ConvInt8 / DepthwiseConvInt8 / Deconvolution / DeconvolutionDepthwise
- **矩阵族（LLM 关键）**：MatMul / BatchMatMul / Attention
- **激活/归一化**：UnaryOp / BinaryOp / Eltwise / Softmax / LayerNorm / InstanceNorm / LRN / Scale / Cast
- **布局变换**：Reshape / Transpose / Permute / Concat / Slice / SliceTf / StridedSlice / Pack / Squeeze /
  Unsqueeze / ExpandDims / Flatten / Crop / ConvertTensor / Tile / Padding / Interp / Pooling / Pooling3D / DepthToSpace
- **量化桥**：FloatToInt8 / Int8ToFloat / EltwiseInt8
- **其他**：GatherV2 / ArgMax / TopKV2 / NonMaxSuppressionV2 / Reduction / Rank / Shape / Identity / BroadcastTo / ...

`NPUBackend::onCreate()`（`NPUBackend.cpp:568`）：查表 → 找到则构造执行器；**找不到（或构造器返回
nullptr）则返回 nullptr，由调度层把该 op 落到 CPU**（见 §6.1）。

### 3.3 MNN 图 → NPU 图：一次性在线构图

关键数据结构（`NPUBackend.hpp`）：

```cpp
map<int, vector<pair<shared_ptr<ge::Operator>, string>>> mGrapMap;   // tensorIndex → 该 tensor 的 ge::Operator 链
vector<pair<shared_ptr<ge::Operator>, MNNTensorList>> mOutGEOpMap;   // 终端输出 op + 对应 MNN OUTPUT tensor
map<int, std::vector<ge::Operator>> mInputOps;                       // 网络输入 Data op（按 inputIndex）
map<unsigned long, int> mInputMap;                                   // MNN 输入 tensor 指针 → HiAI 输入槽位
std::map<Backend::MemObj*, int> mOutputMemToIndex;                   // MNN 输出 MemObj → HiAI 输出槽位
```

构图发生在 **`onResize`（会话 resize）阶段**，分三步，所有 NPU op 共享同一套流程：

#### 第一步：`setNetworkInput()`（`NPUBackend.cpp:366`）

对当前 op 的每个输入 tensor：
- **usage == INPUT**：创建一个 `hiai::op::Data(opName)`，用 `ge::TensorDesc(Shape, FORMAT_NCHW, DT_FLOAT)`
  描述 shape（NHWC 输入会标成 `FORMAT_NHWC`；int32/int64 输入改数据类型），记入 `mInputOps` 与 `mGrapMap`；
  - 本地修复：HiAI DDK 拒绝 rank > 4 的 Data op，因此对 5D+ 输入（如 Qwen3-VL 的
    rotary_pos_emb `[2,1,S,1,D]`）先**挤压单位维**使 rank ≤ 4，并把被挤掉的维位置记到
    `mInputSqueezedAxes`，供下游 op 补偿（`NPUBackend.cpp:420-449`）。
- **usage == CONSTANT**：创建一个 `hiai::op::Const(opName)`，把 host 内存的权重/偏置通过
  `ge::Tensor::SetData()` 拷进 Const op（权重直接内嵌进 IR 图，不参与运行时传输）。

#### 第二步：`getInputOps()`（`NPUBackend.cpp:1163`）

按 `op->inputIndexes()` 在 `mGrapMap` 中查找**上游 producer 的 ge::Operator**，返回本 op 需要的输入算子。
这就是图边（edge）的建立方式 —— MNN 的 tensorIndex 通过 `mGrapMap` 桥接成 ge 算子间的连线。

#### 第三步：每个 op 自己搭 ge 子图，再 `setOutputOps()`

每个 `NPUXxxExecution::onResize()` 做三件事（以 `NPUConvolution.cpp:41` 为例）：

```cpp
mNpuBackend->setNetworkInput(inputs, mOp);   // 补建 Data/Const
auto xOp = mNpuBackend->getInputOps(mOp);    // 拿上游 ge 算子
// ... 根据 Op 参数构造 hiai::op::Xxx 链：
//     hiai::op::Convolution  / hiai::op::MatMul（1x1 Linear 特判，走 Da Vinci CUBE）
//     mConst_w / mConst_b（hiai::op::Const，权重经 set_attr_value 内嵌）
// ... 把输出链上每个算子 SetInput(xOp) 串联
mNpuBackend->setOutputOps(mOp, HIAI_op, outputs);  // 登记输出映射
```

`setOutputOps()`（`NPUBackend.cpp:1200`）：
- 把本 op 的 HIAI_op 链按 `outputIndexes()` 全部登记进 `mGrapMap`（供下游消费）；
- 对 usage == OUTPUT 的 tensor，把（最后一个 ge 算子, 输出 tensor 列表）追加进 `mOutGEOpMap`，
  它们是后续 `graph.SetOutputs()` 的集合。

**NPUAttention 示例**（`NPUAttention.cpp:38`）：Q/K/V → `Permute`（转成 BNHS? 布局）→
Q×K `BatchMatMul` → `Mul`(scale) → `Add`(mask) → `Softmax` → ×V `BatchMatMul` → `Permute` 回，
完全在 ge 图中手工搭建 attention 计算链。

**格式转换辅助**（`NPUBackend.hpp` 内联函数）：
- `tensorShapeFormat()` / `convertShape()`：把 MNN 任意 rank（1~8D）映射成 NPU 4D NCHW/NHWC；
  >4D 时合并到 N 维；NHWC 时按 `0,2,3,1` 重排；
- `axisFormat()`：把 MNN 语义的 axis 换算成 NCHW 语义（NHWC 布局时轴映射表）。

#### 第四步：`onResizeEnd()` → `bulidIRModelAndLoad()`（`NPUBackend.cpp:813 / 976`）

构图完成后，整个 `ge` 图一次性编译加载，阶段如下：

```
ge::Graph graph("Graph1");
graph.SetInputs(mInputOps 的算子).SetOutputs(mOutGEOpMap 的算子);   // 图 I/O
ge::Model model("0", "model_v000011");    // modelName + 版本
model.SetGraph(graph);

domi::HiaiIrBuild ir_build;
ir_build.CreateModelBuff(model, om_model_buff);   // 分配 OM 输出缓冲
ir_build.BuildIRModel(model, om_model_buff);      // ★ IR → OM 二进制（大图 30s~5min）
WriteToOMFile(om_model_buff, "<npu_dir>/vision.om");   // 可选缓存
mMgrClient = LoadModelSync(om_model_buff, modelName);  // 加载进 NPU
getInOutTensorInfo(modelName);                        // 回读 I/O 维数，分配 AiTensor
```

- `HIAI_DEBUG` 打开时额外 `model.Save(buffer)` 把 ge 模型序列化成 `test.irpb` 落盘（离线转 OM 用）；
  平时跳过 Save 以省内存（24 层 ViT 的 Save 序列化峰值可达 3GB+，会被鸿蒙 watchdog OOM kill，
  代码注释中专门说明了这一点）；
- **OM 缓存**（`MNN_HIAI_CACHE_OM_BY_CHUNK=1` 时）：优先从 `<pNPUModelDirPath>/chunk_i/vision.om`
  直接加载跳过 IR 编译；miss 时编译并把 OM 写盘（详见 §5.4）。

#### 第五步：`getInOutTensorInfo()`（`NPUBackend.cpp:832`）

用 `mMgrClient->GetModelIOTensorDim(modelName, ...)` 回读 NPU 侧真实的输入/输出维数，
据此分配 `hiai::AiTensor`（输入/输出各一组，**ION 共享内存，host 可读写**），然后做 MNN↔NPU 的槽位匹配：

- **输入**：`mInputMap[MNN tensor 指针] = HiAI 槽位`。由于 DDK 的输入排序可能与 MNN 的分配顺序不同
  （多输入 op 尤其如此），按**字节大小匹配**：先匹配唯一大小的（精确），再按出现顺序匹配同大小
  （如 attention 的 Q/K/V 大小相同，按 inputIndexes 顺序分配，是 best-effort）；
- **输出**：DDK 可能把多个同 shape 输出合并成一个 buffer（例如 hidden_states 与 deepstack 同源），
  因此同样按字节大小做两遍匹配，并记录 `mOutputMemToIndex[MemObj*] = 槽位`。
  之所以用 `Backend::MemObj*` 而不是 `Tensor*`，是因为 `StaticModule::onForward` 会 `Tensor::clone()`
  产生新的 Tensor 对象，指针对比会失败，而克隆共享底层 MemObj。

### 3.4 运行时执行

`NPUBackend::onExecuteBegin()/onExecuteEnd()`（`NPUBackend.cpp:614-623`）包住整次会话执行：

```cpp
void NPUBackend::onExecuteEnd() const {
    process(0);   // 整个图一次 NPU 推理
}
int NPUBackend::process(int modelIndex) const {
    hiai::AiContext context; context.AddPara("model_name", "0");
    int istamp;
    mMgrClient->Process(context, mInputTensors, mOutputTensors, /*timeout=*/1000, istamp);
}
```

数据搬运由 `onCopyBuffer()`（`NPUBackend.cpp:637`）完成，按 usage 分三种情况：

| 情况 | 行为 |
|---|---|
| 常量（CONSTANT） | 直接把目标 tensor 的 host 指针指到源（图构建期已把权重拷进 ge Const，host 侧可复用） |
| 输入（INPUT） | 按 `mInputMap` 找到槽位，把 MNN tensor 拷进 `AiTensor::GetBuffer()`；用 `MNNCPUCopyBuffer` 安全处理 NC4HW4→NCHW 解包 |
| 输出（OUTPUT） | 按 `mOutputMemToIndex`（或大小兜底）找到槽位，把 `AiTensor` 缓冲拷回 MNN tensor；同样用 `MNNCPUCopyBuffer` 处理 NCHW→NC4HW4 重打包 |

> **要点**：NPU 的 AiTensor 是 host 可见的共享内存，CPU 侧通过 memcpy 直接读写 NPU 输入/输出缓冲，
> 不需要显式的 DMA 管理 —— 这正是"CPU 与 NPU 通过 host 内存协作"的最底层机制。

### 3.5 常量内存优化：`MNN_HIAI_FREE_CONST_HOST`

多 chunk 加载时 host 侧权重常驻内存会推高峰值 RAM。本仓库新增了引用计数机制
（`NPUBackend.hpp:32`、`addConstRef`/`consumeConst`，`NPUBackend.cpp:542-566`）：

- `onCreate` 时对每个 CONSTANT 输入 `addConstRef(tensor)` 计数；
- 每个 op 把自己的权重拷进 `hiai::op::Const` 后调用 `consumeConst(tensor)` 减计数；
- 计数归零 → 释放该 tensor 的 host 内存（`describeOrigin->mem = nullptr`）。
- 代价：要求每个消费 op 严格按 addConstRef 次数 consume；`NPUConvolution` 中专门做了
  "host 内存已被释放时复用已构建的 mConst_w/mConst_b" 的保护（`NPUConvolution.cpp:88-126`）。

### 3.6 量化权重（GPTQ 8-bit）的处理：反量化时机与存储开销

> 本节回答两个问题：**8-bit 权重在哪里被反量化**、**NPU 上的权重存储开销是多少**。
> 结论先行：**反量化发生在编图期（onResize），由宿主 CPU 在进程内存里一次性完成，产物是 fp32
> （不是 fp16）；NPU 侧 OM 的权重段默认也以 FP32（4B/权重）存储，fp16 只存在于 CUBE 计算管线的
> SRAM 中，存储精度 fp32、计算精度 fp16**。

#### 3.6.1 存储格式：per-group，而非 per-tensor / per-channel

GPTQ w8g128 导出的权重是 **per-group 非对称量化**（`mnn_converter.py:432-479` 落盘格式）：

- 每个权重张量 = int8 blob + alpha（每组 **[zero, scale] 两个 fp32**）+ bias，
  external 布局 `[weight_offset, weight_len, alpha_len, bias_len, 0]`；
- 实证（`visual_blocks_npu_0.mnn.json`）：q_proj 1024×1024 → alpha_len=65536B → 16384 个 fp32
  = 1024×(1024/128) 组 × 2；mlp 4096×1024 → 262144B → 65536 = 4096×(1024/128)×2；
- quanParameter 关键字段：`type=1`（IDST）、`aMaxOrBits=8`、`aMin=1`、`readType=8192`
  （≠0 → `asymmetric=true`，`outputCount = alphaSize/2`；`aMin=1>0` 不触发 clampMin 修正）。

#### 3.6.2 反量化位置与时机：编图期、宿主 CPU、一次性

全图路径下 Conv op 由 `NPUConvolution` 执行（`NPUConvolution.cpp:271` 注册 `OpType_Convolution`），
其 `onResize`（编图阶段）对带 `quanParameter` 的卷积**无条件反量化**（`NPUConvolution.cpp:79-85`）：

```cpp
quanCommon = ConvolutionCommon::load(mOp, backend(), true);  // forceFloat=true
filterDataPtr = quanCommon->weightFloat.get();               // fp32！
```

`ConvolutionCommon::load(forceFloat=true)`（`ConvolutionCommon.cpp:557+`）流程：

1. 按 `external[0]` 偏移从 `.weight` 文件读 int8 blob + alpha；
2. 反量化循环（`ConvolutionCommon.cpp:806-826`）：
   `dstW[v] = (float)srcW[v] * alpha[2o+1] + alpha[2o]`（scale×int8 + zero，按 group 索引）；
3. 反量化后 `weight.release(); alpha.release()` —— int8 副本与 alpha 立即释放，只有 fp32 存活。

即：**int8 权重从未离开宿主进程；反量化是纯 CPU 堆内存计算，每个权重张量只在编图期做一次，
推理期零反量化**。反量化后的 fp32 被 `SetData` 封进 `ge::Tensor`（`DT_FLOAT`）→ `ge::op::Const`
→ `mGrapMap` → `bulidIRModelAndLoad` 组装整 chunk 图 → `BuildIRModel` 编译成 OM。

#### 3.6.3 为什么用不上 NPU 的 int8 计算：per-channel 门槛

HiAI 真 int8 算子（`QuantizedConvolution` / `QuantizedMatMul` / `QuantizedFullyConnection`）要求
**per-channel** scale（`alpha.size() == outputCount`，`HiAIConvExecution.cpp:566`，取 scale 只取前 OC 个）。
group alpha（16384 ≠ 1024）不满足 → 量化分支整体跳过 → 永远走 fp32 反量化路径。因此：

- 全图路径 `NPUConvolution.cpp:79` 根本没有 per-channel 检查，直接 forceFloat 加载，也无
  `HIAI_CONV_QUANT` 之类开关；
- delegate 路径（`HiAIConvExecution`）虽有 `HIAI_CONV_QUANT=full/matmul_int8/fc_int8` 分支，
  但 group 量化不满足 per-channel 条件，这些开关**对 group 模型无任何效果**；
- `NPUConvolutionInt8`（`NPUConvolutionInt8.cpp:175` 注册 `OpType_ConvInt8`）走的是旧式
  `symmetricQuan()` 格式（per-channel），与 `quanParameter` 无关，同样不适用。

#### 3.6.4 NPU 侧存储开销：OM 权重段默认 FP32（不是 fp16）

HiAI DDK 的 `BuildOptions`（`3rdParty/include/compatible/HiAiModelBuilderType.h:24,43-55`）提供
权重数据类型选项，**默认是 FP32**：

```cpp
enum WeightDataType { FP32, FP16 };
struct BuildOptions {
    ...
    WeightDataType weightDataType = FP32;  // ← 默认 FP32
    ...
};
```

而 MNN 的两处编译调用（`NPUBackend.cpp:1073,1082` 全图路径；`HiAIConvExecution.cpp:1090,1096`
delegate 路径）都是 **2 参数版** `CreateModelBuff(model, buf)` / `BuildIRModel(model, buf)`，
不传 BuildOptions → 走 `hiai_ir_build.h:30` 的 `defaultBuildOptions`（默认构造 → FP32）。
**因此 OM 权重段以 FP32（4B/权重）存进 NPU 权重内存**；fp16 只出现在 CUBE 计算时刻
（注释证据：`HiAIConvExecution.cpp:835` "NPU dequants weight to fp16 before CUBE MAC"），
是**存储精度 fp32、计算精度 fp16**。

权重存储的完整账本：

| 阶段 | 位置 | 精度 | 单权重字节 |
|---|---|---|---|
| 磁盘模型 | `.weight` external | int8 + per-group [zero,scale] | ≈ 1.06 B |
| 编图期临时 | 宿主进程 RAM | fp32（int8 副本反量化后即释放） | 4 B |
| **OM 权重段** | **NPU 权重内存（DRAM）** | **FP32（默认 BuildOptions）** | **4 B** |
| 计算时刻 | CUBE SRAM | fp16 MAC（硬件转换，仅存在于计算管线） | 不占存储 |

实际影响：

- 4 个 NPU chunk ≈ 4×53.7MB int8 → **OM 权重存储约 860MB fp32**（int8 源文件的 4 倍），
  8-bit 量化在 NPU 上只省了磁盘与加载带宽，运行时存储/算力无收益；
- **潜在 buffer 边界风险**：`CreateModelBuff` 默认分配 200M 输出缓冲（`hiai_ir_build.h:36`），
  单 chunk fp32 OM ≈ 215MB 已接近上限。若真机报 buffer 不足，需改用 3 参数版
  `CreateModelBuff(model, buf, 0)`（customSize=0 自动计算）或
  `BuildIRModel(model, buf, {weightDataType=FP16})` 传 BuildOptions —— 均涉及代码改动
  （`NPUBackend.cpp:1073/1082`）。

#### 3.6.5 对照：CPU 侧才是"运行时按需反量化"

同样的 8-bit 权重在 CPU 上（llm.mnn 主体 + CPU chunk）走 `OpType_ConvInt8`（`CPUBackend.cpp:650`）
→ `ConvInt8TiledExecutor`（支持 per-group [zero, scale]，`ConvInt8TiledExecutor.cpp:215-261`）：
**int8 权重保持 int8 留在内存**，计算中 int8×int8 MAC → int32 累加 → **在 SIMD 寄存器/一级缓存里
按 group scale 反量化输出**。这才是发生在 cache/寄存器中的反量化；NPU 侧则完全没有运行时反量化。

---

### 3.7 从 HF 模型到 NPU 图：完整转换链路（以 attention 为例）

前面的章节从运行时角度讲"MNN 图 → ge 图 → OM"；本节把镜头拉远，讲清楚 README §2.3
导出命令背后的**整条转换链路**：一个 HuggingFace 模型如何一步步变成 NPU 上执行的图。
以 attention 算子作为贯穿示例——它在链路的每一阶段形态都不同，是最能说明问题的一个算子。

#### 3.7.1 链路全景

```
HF torch 模型定义（transformers/llm/export/）
  │ ① torch.onnx.export（llmexport.py）
ONNX 图 ── LLM 侧：attention 是「LlmExporter::FusedAttention」自定义节点
        └─ 视觉侧：attention 是 MatMul/Softmax/Add 展开算子
  │ ② MNNConvert --transformerFuse --allowCustomOp（mnn_converter.py）
MNN 图（融合后的单个 OpType_Attention，权重存外部 .weight 文件）
  │ ③ NPUBackend::onCreate 设备端在线构图（NPUAttention.cpp）
ge 图（attention 被重新展开为 10 个 HiAI 基础算子）
  │ ④ CreateModelBuff + BuildIRModel（设备端在线 IR 编译）
OM（离线模型，权重段 FP32 烘焙，见 §3.6）
  │ ⑤ LoadModelSync + Process()
NPU 推理
```

本仓库模型的实际产物：语言模型 `llm.mnn`（28 层，每层 1 个 attention，**kv_cache=true**），
视觉块 `visual_blocks_npu_0..5.mnn`（24 块，每块 1 个 attention，**kv_cache=false**）。

#### 3.7.2 第 1 步：HF 模型定义——两种 attention 形态

**语言模型**：导出器用 `utils/transformers.py` 的 `Attention` 包装类（`transformers.py:57`）
包住原始 HF attention，`ModelMapper.do_map` 完成权重映射（q_proj/k_proj/v_proj/o_proj）。
其 `forward` 有**两条路径**（`transformers.py:236-284`）：

```python
if self.export_fused_attn and torch.onnx.is_in_onnx_export():
    attn_output = self.fused_attn(query_states, key_states, value_states, attention_mask)
else:
    # 标准展开：matmul(q,k)*attn_scaling → mask → softmax(fp32) → matmul(attn,v) → o_proj
```

**视觉块**：不走包装类——`utils/model.py:365-387` 里 `self.blocks[i]` 是**原始 HF
vision block 对象**（`torch.nn.ModuleList`），直接 `blocks[i](hidden, rotary, mask)`
前向。因此视觉块的 attention 在 torch 层面永远是展开的分解计算。

#### 3.7.3 第 2 步：ONNX 导出——LLM 用自定义节点，视觉用展开算子

**LLM（fused 路径）**：`utils/custom_op.py` 的 `FusedAttentionOp.symbolic` 把整个
attention 计算导出为**一个自定义 ONNX 节点**：

```python
return g.op("LlmExporter::FusedAttention", query, key, value, attention_mask,
            output_dim_i=output_dim, kv_cache_i=kv_cache,
            name_s=name, layer_index_i=layer_index,
            kv_shared_layer_index_i=kv_shared_layer_index)
```

即 ONNX 图里 Q/K/V/mask 四路输入直接汇聚到一个 `LlmExporter::FusedAttention` 节点，
参数全部打进属性（forward 只返回 zeros 占位，导出时不会真的算一遍）。

**视觉（`--visual_split`）**：`llmexport.py:export_vision_split()` 把视觉编码器切成
pre / blocks / post 三个 ONNX 子图分别导出。blocks 的输入输出都是 3D `[B, S, D]`，
chunk 边界无需 NC4HW4 布局转换（这也是它能整块上 NPU 的前提，见 §5.2）。这里的
attention 在 ONNX 里就是常规的分解算子：`MatMul → Mul(scale) → Add(mask) →
Softmax → MatMul(attn, V)`。

#### 3.7.4 第 3 步：ONNX → MNN（MNNConvert）——两条路都汇合到 `OpType_Attention`

`utils/mnn_converter.py:onnx2mnn()` 调 MNNConvert 时默认带
`--transformerFuse --allowCustomOp --saveExternalData --weightQuantAsymmetric=0`
（`mnn_converter.py:70-87`）。注意此处的"Fuse"与 §3.6 的量化无关——**`--transformerFuse`
是开启 converter 端把分解的 transformer 算子重新融合成高级 op**。

**视觉路径（C++ 模板融合）**：`tools/converter/source/optimizer/merge/FuseAttention.cpp`
注册了一个 `TemplateMerge` 的 "FuseAttention" pass。匹配阶段从尾部 Reshape 逆推
`Reshape → Transpose → MatMul(attn@v) → Softmax → (mask 的 Select/BinaryAdd，含
sinks 的 BinarySub→Concat 路径) → BinaryOp → MatMul(qk) → Transpose → query/key`
整条链（GQA 模型额外处理 `BroadcastTo/Unsqueeze`）；fold 阶段把它替换成**单个
`OpType_Attention`**：

```cpp
attention->name       = "Attention" + expr->name();   // FuseAttention.cpp:177
attention->type       = OpType_Attention;
param->kv_cache       = kvcache;                       // 视觉 = false
VARPS inputs = {query, key, value, mask};
...
Expr::replace(expr, attention_expr->expr().first);    // 整条链被一个节点取代
MNN_PRINT("Fuse Attention as %s [kvcache: %d, has_sinks: %d]\n", ...);
```

**LLM 路径（Python json 层重写）**：`LlmExporter::FusedAttention` 自定义节点被
`--allowCustomOp` 保留为 Custom op 进入 MNN 图；导出器随后
`mnn2json → rebuild_attnention → json2mnn`（`mnn_converter.py:543-572`），把 Custom op
的 attr 数组解析后**重写为 `type: "Attention"`、`main_type: "AttentionParam"`、
`main: {kv_cache: bool, layer_index, kv_shared_layer_index}`**。视觉 chunk 的 GPTQ 8-bit
量化也在这条 mnn2json→json2mnn 管道里完成（§3.6 已述）。

#### 3.7.5 第 4 步（实测证据）：MNN 图里的 Attention op 长什么样

用 MNNConvert 把 `visual_blocks_npu_0.mnn` 转回 json，视觉块里 attention 的最终形态：

```json
{ "name": "Attention/blocks.0/self_attn/Reshape_3_output_0",
  "type": "Attention",
  "inputIndexes": [87, 120, 131, 132],
  "main": { "kv_cache": false, "layer_index": -1,
            "kv_shared_layer_index": -1, "output_c4": false, "attnScale": 0.0 } }
```

- **名字** `Attention/blocks.0/self_attn/Reshape_3_output_0` 正是 FuseAttention 的
  `"Attention" + expr->name()`——被替换子图的入口 Reshape 节点名；
- **inputs[0..2]** = Q / K / V：分别来自 `q_proj/k_proj/v_proj` 的 `Add_output_0`
  经 Reshape 链——说明 **Linear 投影已被折叠成 1×1 Convolution** 留在融合算子外面，
  fusion 只吃掉"投影之后"的核心 attention 计算；
- **inputs[3]** = `attention_mask`（整 chunk 共享的 Input）；
- **kv_cache=false**：视觉是双向 encoder attention，无 KV 缓存。

每块 1 个（chunk 0 的 689 个 op 中含 4 个 Attention），6 chunk 共 24 个；llm.mnn 侧
28 个（kv_cache=true，经 rebuild 路径写入）。

#### 3.7.6 第 5 步：MNN Attention → ge 图（NPUAttention.cpp）

`OpType_Attention` 在 NPU 侧注册了 `NPUAttention`（`NPUAttention.cpp:266`）。注意：
**NPU 侧并没有使用 `ge::op::Attention`**——因为 MNN 的 fused Attention 只含核心
attention 计算（Q/K/V 投影已在外层以 1×1 Conv 完成），与 HiAI 面向完整 transformer
层的 Attention 算子参数语义对不上；MNN 的做法是在 `onResize` 里把融合算子**重新展开**
为 10 个 HiAI 基础算子（`NPUAttention.cpp:60-262`）：

| # | ge 算子 | 作用 |
|---|---------|------|
| 1 | `Permute` ×3 | Q/K/V：`[B,S,H,D] → [B,H,S,D]` |
| 2 | `Const` + `Mul` | Q 乘 `1/√head_dim`（scale；MNN 侧 `attnScale=0.0` 未启用，故固定用 1/√d） |
| 3 | `BatchMatMul` | QKᵀ → `[B,H,Sq,Skv]` |
| 4 | `Reshape` + `Add` | mask `[B,Sq,Skv] → [B,1,Sq,Skv]` 后加到 QKᵀ（显式升到 rank 4，规避 DDK 隐式广播出 5D 的兼容问题，`NPUAttention.cpp:155-163`） |
| 5 | `Softmax(axis=-1)` | 沿 kv 维做 softmax |
| 6 | `BatchMatMul` | attn × V → `[B,H,Sq,D]` |
| 7 | `Permute` | 回到 `[B,Sq,H,D]` |
| 8 | `Reshape` | 展平为 `[B,Sq,H*D]`（`NPUAttention.cpp:235`） |

最后 `setOutputOps(mOp, chain, outputs)` 把整条链登记进 mGrapMap（`NPUAttention.cpp:254`），
后续与 §3.3 完全一致：`bulidIRModelAndLoad()` 汇总成 `ge::Graph` → `CreateModelBuff` +
`BuildIRModel` 设备端在线编译成 OM（权重段 FP32，见 §3.6）→ `LoadModelSync` → `Process()`。
每条 `_q_perm` / `_qk` / `_softmax` 等中间算子名都带 `opName` 前缀，保证 chunk 内不重名。

#### 3.7.7 各阶段形态对照表

| 阶段 | attention 形态 | 证据 |
|------|---------------|------|
| HF torch 定义 | 语言模型：`Attention` 包装类双路径；视觉：原始 eager 展开 | `transformers.py:57,236`；`model.py:365-387` |
| ONNX（LLM） | 单个 `LlmExporter::FusedAttention` 自定义节点 | `custom_op.py:38-51` |
| ONNX（视觉） | `MatMul/Mul/Add/Softmax` 分解子图 | `export_vision_split()` |
| MNN（视觉） | 融合 `OpType_Attention`（kv_cache=false） | FuseAttention.cpp:184-191；mnn.json 实测 |
| MNN（LLM） | 融合 `OpType_Attention`（kv_cache=true） | mnn_converter.py:543-572 rebuild_attnention |
| ge 图 | 10 个基础算子（Permute/Mul/BatchMatMul/Softmax/Add/Reshape） | NPUAttention.cpp:60-262 |
| OM | 设备端编译产物（fp32 权重烘焙） | §3.3/§3.6 |

一句话：**HF 的展开 attention →（视觉）converter 模板融合 /（LLM）自定义节点 + json
重写 → MNN 单个融合 `OpType_Attention` →（NPU 侧）重新展开成 HiAI 基础算子链 →
在线编译成 OM**。融合发生在 MNN 图，展开发生在 ge 图，两次形态变化互为逆过程，
中间产物（MNN 图）恰好是跨后端通用的稳定表示。

---

## 4. 路径②：HiAI 逐算子委托（`MNN_FORWARD_USER_1` / `hiai_delegate`）

### 4.1 设计

`HiAIDelegateBackend` **继承 `CPUBackend`**（`HiAIDelegateBackend.cpp:30-33`），
即会话主执行者还是 CPU，只是 `onCreate()` 拦截 `OpType_Convolution`：

```cpp
if (op->type() == OpType_Convolution) {
    // 上限 70 个：每个 AiModelMngerClient + 已加载模型都占用 NPU 驱动资源
    // （RPC fd、设备内存、固件槽位），超过上限 NPU 会 Load 失败
    static constexpr int kMaxHiAIConvs = 70;
    static std::atomic<int> sHiAIConvCount{0};
    if (sHiAIConvCount.fetch_add(1) >= kMaxHiAIConvs)
        return CPUBackend::onCreate(inputs, outputs, op);
    return new HiAIConvExecution(this, op, inputs, outputs);
}
return CPUBackend::onCreate(inputs, outputs, op);   // 其他 op 原样走 CPU
```

运行时创建时同样自检：`AiModelMngerClient::Init + GetVersion()` 可用才注册（`HiAIDelegateBackend.cpp:54-74`）。

### 4.2 每个卷积独立编译一个小 HiAI 模型

`HiAIConvExecution::compileHiAIModel()`（`HiAIConvExecution.cpp:462`）在 `onResize` 里为单个卷积
构造 ge 图并 `BuildIRModel` + `LoadModelSync`，随后**预分配** AiTensor I/O 并在 `onExecute` 中复用
（`AiTensor::Init()` 的 ION 分配是每次调用的大头，故只做一次）。

执行路径选择（`HiAIConvExecution.cpp:514-620`，支持 env 覆盖做 A/B）：

| 路径 | 条件/开关 | 图 | 说明 |
|---|---|---|---|
| `MatMul` | 1×1 conv 伪装 Linear（`isMatMulConvertedConv`） | `hiai::op::MatMul` | Linear 在推理时被导出为 Reshape→Conv1x1→Reshape；NPU 上 MatMul 直接走 Da Vinci CUBE，远快于卷积引擎 |
| `QuantizedConvolution` | `HIAI_CONV_QUANT=full` | int8 filter + per-channel scale | 真 int8×int8 CUBE MAC（`x_quant_type=1`）；`HIAI_INT8_X_SCALE` env 控制输入量化 scale |
| `QuantizedConvolution` | 默认量化 | int8 weight + fp16 MAC | weight-only 量化 |
| `QuantizedMatMul` | `HIAI_CONV_QUANT=matmul_int8` | 单 `hiai::op::QuantizedMatMul` | 图边界 fp32，NPU 内 int8×int8→int32→per-channel rescale→fp32；要求固件 ≥ 100.500.010.010 |
| `QuantizedFullyConnection` | `HIAI_CONV_QUANT=fc_int8` | `hiai::op::QuantizedFullyConnection` | 同上变体 |
| 普通 `Convolution` | 默认 | `hiai::op::Convolution` | fp32 |

编译失败有**诊断探针**（`probeQuantizedMatMul` 等，`HiAIConvExecution.cpp:104-460`）：用最小子图逐个
probe QuantizedMatMul 的 per-tensor / per-channel / +bias 组合，定位 DDK 拒绝的 attr；
int8 主路径失败时**自动重试 dequant→fp32 路径**，再失败才回退 CPU（`HiAIConvExecution.cpp:1230-1258`）。

### 4.3 执行三阶段

`onExecute()`（`HiAIConvExecution.cpp:1262`）：

```
1) 上传输入：NC4HW4 → MNNCPUCopyBuffer 直接解包进 AiTensor 缓冲；否则 memcpy
2) NPU 推理：mMgrClient->Process(mContext, mHiAIInputs, mHiAIOutputs, 1000, stamp)
   （AiContext 与 AiTensor 均缓存，无每帧分配）
3) 拉回输出：AiTensor 缓冲 → memcpy / MNNCPUCopyBuffer 重打包回 MNN tensor
```

---

## 5. LLM（Omni）中的 NPU 编排

### 5.1 后端字符串 → ForwardType

```cpp
// omni.cpp:232 / llm.cpp:41
"npu"           → MNN_FORWARD_NN        // 高通 QNN（另一套后端）
"hiai"          → MNN_FORWARD_USER_0    // 路径①
"hiai_delegate" → MNN_FORWARD_USER_1    // 路径②
```

### 5.2 视觉模型拆分：pre / blocks / post

`Omni::load()`（`omni.cpp:310-500`）中，当 `visual_split=true` 时：

- `visual_pre_model` / `visual_post_model`：加载到 `mProcessorRuntimeManager`（通常 CPU，`shapeMutable=true`）；
- `visual_blocks_*`：加载到独立的 `mVisionBlocksRuntimeManager`（典型为 HiAI/NPU），
  **`shapeMutable=false` + `rearrange=false`** —— NPU 图在编译时定形，运行期不允许动态 shape。

blocks 有三种加载模式：

```
(a) visual_blocks_chunks = [a.mnn, b.mnn, ...]     ← 推荐：K-chunk 拆分
    │ 每个 chunk 是更小的 .mnn；顺序加载使 HiAI IR 编译峰值内存降到整图的 1/K
    │ 可选 visual_blocks_chunk_backends = ["npu","cpu",...] 逐 chunk 指定后端
    │ 可选 visual_blocks_om_paths[i]：该 chunk 用预编译 .om 直通（路径③，mChunkUseOm[i]=true）
(b) visual_npu_layers = N > 0                        ← 兼容：旧 2-chunk
    │ visual_blocks_npu_model（前 N 层，NPU）+ visual_blocks_cpu_model（其余，CPU）
(c) 默认：整个 visual_blocks.mnn 单块跑 NPU runtime
```

### 5.3 chunk 执行循环（CPU↔NPU 流水协作）

`Omni::visualForward` 的 chunk 循环（`omni.cpp:915-1010`）：每个 chunk 可能走两种执行器，逐 chunk 串行：

```
MNN chunk:  mVisionBlocksChunkModules[i]->onForward({curHidden, rotary, mask})
            └─ 输出经 makeHostBridgeVar() 强制物化到 host 内存，避免 NPU chunk 输出留在设备侧

OM chunk:   varpToFloatVector(curHidden/rotary/mask)          // VARP → std::vector<float>
            mNpuChunkExecutor->runChunk(i, hidden, rotary, mask, omOutputs)
            floatVectorToVarp(omOutputs[0], dims)             // float → VARP（NCHW _Input）
            (void)nextHidden->readMap<void>();                // 强制同步回 host
```

- 前一个 chunk 的输出 `curHidden` 是下一个 chunk 的输入，**桥接层全部是 host 浮点数组**；
- `visual_blocks_om_deepstack_dup` 配置：OM 模型把 hidden==deepstack 合并成一个输出时，
  在 MNN 侧把输出 0 复制一份当作 deepstack（`omni.cpp:927-975`）。

### 5.4 chunk 是如何拆分的（按层等分）

chunk 拆分发生在**导出端**（`transformers/llm/export/llmexport.py`），运行时只是按序消费。
拆分的维度是**视觉 Transformer 的 block（层）**，不是按算子或按层内结构拆分。

入口参数（`llmexport.py:1352-1370`）：

```
--visual_split             # 视觉模型拆成 visual_pre.mnn / visual_blocks.mnn / visual_post.mnn
--visual_npu_chunks K      # 把 blocks 等分成 K 个 chunk：visual_blocks_npu_0.mnn ... _{K-1}.mnn
                           # 每个 chunk 的权重是整块的 1/K → HiAI IR-build 峰值内存降为 1/K
                           # （整块编译会 OOM 时用这个）。K 必须在 [2, total_blocks] 之间。
--visual_chunk_backends    # 可选逐 chunk 后端路由："npu,cpu,npu,..."，默认全 NPU
--visual_npu_layers N      # legacy 2-chunk：前 N 层 → visual_blocks_npu.mnn（NPU），
                           # 其余 → visual_blocks_cpu.mnn（CPU）。与 K-chunk 互斥，K 优先。
```

等分算法（`llmexport.py:955-965`）：

```python
total = len(self.visual.blocks)         # 如 24 层 ViT
base = total // npu_chunks              # 每 chunk 基础层数
rem  = total % npu_chunks               # 余数层
cursor = 0
for ci in range(npu_chunks):
    size = base + (1 if ci < rem else 0)   # 前 rem 个 chunk 各多 1 层
    s, e = cursor, cursor + size
    chunk_specs.append((s, e, f'visual_blocks_npu_{ci}.onnx', False))
    cursor = e
```

即：**chunk 大小相差不超过 1 层**。例如 24 层 / `--visual_npu_chunks 4` → 每个 chunk 6 层；
25 层 / 4 chunks → 7/6/6/6 层。

每个 chunk 由 `_build_visual_blocks_chunk(s, e, ...)`（`llmexport.py:748`）构建：
`nn.ModuleList(vis.blocks[s:e])`，即**连续的层切片**，I/O 签名与整块完全一致：

```
输入: (hidden_states, rotary_pos_emb, attention_mask)
输出: (hidden_states [, deepstack_hidden_<全局索引>...])
```

要点：
- **deepstack 层按全局 layer index 归属所在 chunk**；输出名使用全局索引
  （`deepstack_hidden_0..M-1`），保证 post 模块直接消费，chunk 拼接后顺序不变；
- 相邻 chunk 之间用 **dry-run 传播 hidden_states** 作为下一个 chunk 的 ONNX 追踪输入
  （ViT 层形状不变，只是修正输入数值）；
- 每个 chunk 独立 ONNX → 独立 MNN 转换（同一套量化参数；GPTQ 路径用 `layer_offset`
  把 chunk 内局部层号映射回全局层号查 safetensor key）；
- 运行时（omni.cpp）按 `visual_blocks_chunks` 数组顺序串行执行，chunk 间用 host 浮点数组桥接
  （见 §5.3）。

> **关于"OM 大小限制"**：仓库代码里没有硬编码的 OM 文件大小上限（如 200MB）——没有这样的常量或
> 校验。代码注释中反复出现的拆 chunk 动机是两点：① **HiAI IR-build 峰值内存（RAM）**——整块 24 层
> ViT 的 `model.Save()` 序列化峰值可达 3GB+，会被鸿蒙 watchdog OOM kill（`NPUBackend.cpp:1052-1057`），
> `BuildIRModel` 大图耗时 30s~5min；② 拆分后每 chunk 的 IR 编译只处理 1/K 的权重，峰值内存与编译时长
> 线性下降（"Recommended when the monolithic build OOMs"）。若你在实际设备上观察到 OM 加载失败与
> 大小相关（如 200MB 量级），那是 **HiAI DDK / NPU 驱动的外部限制**（单模型共享内存、RPC 传输、
> 固件槽位等），不在本仓库代码内——拆 chunk 正是规避该限制的手段之一。

### 5.5 OM 缓存的落地：`EXTERNAL_NPU_FILE_DIR`

`MNN_HIAI_CACHE_OM_BY_CHUNK=1` 时，chunk 的首次编译产物可缓存：

```cpp
// omni.cpp:444-470
mVisionBlocksRuntimeManager->setExternalPath(chunkNpuDir, MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
// 路径：RuntimeManager::setExternalPath (Executor.cpp:248) → mNpuDir (RuntimeAttr.hpp:20)
//     → StaticModule (StaticModule.cpp:395) → backend->pNPUModelDirPath (Backend.hpp:278)
//     → NPUBackend::bulidIRModelAndLoad → "<dir>/chunk_i/vision.om" 读写
```

即：每个 chunk 的 OM 缓存到 `npu_model_dir/chunk_<i>/vision.om`，模型名取 `vision_chunk_<i>`。

---

## 6. CPU 与 NPU 的协作机制总结

### 6.1 调度层：主后端优先，CPU 兜底

MNN 的会话调度（`express/module/StaticModule.cpp`）对每个 op 先问主后端、失败问 backup：

```cpp
// StaticModule.cpp:147-150（卷积等），193-195（Attention）
exe.reset(backend->onCreate({}, {}, op));                       // 主后端（NPU）
if (exe.get() == nullptr)
    exe.reset(backupBackend->onCreate({}, {}, op));            // backup（CPU）
if (nullptr == exe) break;                                      // 两边都失败才跳过
```

因此**同一张 MNN 图 = 一张 NPU 子图 + 若干 CPU 残差 op**，二者在同一 Session 内共存；
NPU 子图边界处的 tensor 通过 host 内存交换数据。

### 6.2 数据层：host 可见的 ION 内存 + memcpy/格式转换

- NPU 侧 AiTensor 由 DDK 分配，`GetBuffer()` 返回 host 可读写的共享内存指针；
- 输入：CPU 计算结果 memcpy（必要时 NC4HW4→NCHW 解包）进 NPU 输入缓冲；
- 输出：NPU 输出缓冲 memcpy（必要时 NCHW→NC4HW4 重打包）回 MNN tensor；
- 格式转换函数（`NPUBackend.cpp:89-175`）：`MNNPackC4` / `MNNUnpackC4` / `NHWC2NCHW` / `NCHW2NHWC`
  及 Uint8 变体；op 边界统一走 `MNNCPUCopyBuffer` 保证安全性。

### 6.3 形状语义层：静态 shape

- NPU 图在 onResize 时按当前 shape 编译，运行期不可变 → Module 配置 `shapeMutable=false`；
- shape 变化会触发 `onResizeBegin`（清空全部图状态并 `UnLoadModel`）→ 重新构图/编译/加载
  （`NPUBackend.cpp:787-811`）；
- 输入 rank > 4 的 5D 输入会被挤压单位维以通过 DDK 4D 限制，并由 `mInputSqueezedAxes` 记录补偿。

### 6.4 端到端时序（路径①，单次会话）

```
Session/Module 加载 (shapeMutable=false)
  │
  ├─ 调度：每 op → NPUBackend::onCreate（查 Creator 表）
  │         ├─ 支持 → NPUXxxExecution（构图期记录）
  │         └─ 不支持 → nullptr → CPU 兜底
  │
  ├─ onResizeBegin ─ 清空 mGrapMap/mInputOps/...，UnLoadModel
  ├─ 每 op onResize ─ setNetworkInput(建 Data/Const) → getInputOps(连上游)
  │                    → 搭 hiai::op::Xxx 子图 → setOutputOps(登记输出)
  ├─ onResizeEnd   ─ bulidIRModelAndLoad():
  │                    ge::Graph → ge::Model → HiaiIrBuild(BuildIRModel)
  │                    → LoadModelSync → GetModelIOTensorDim → 分配 AiTensor → 槽位匹配
  │
  └─ runSession（循环推理）
       ├─ onCopyBuffer(INPUT)   CPU→NPU（memcpy / 解包）
       ├─ onExecuteBegin → process(0) → NPU 整图推理（AiContext + Process）
       ├─ onExecuteEnd
       └─ onCopyBuffer(OUTPUT)  NPU→CPU（memcpy / 重打包）
```

---

## 7. 编译、部署与离线 OM 生成

1. **编译**：`cmake -DMNN_NPU=ON`（鸿蒙 OHOS 时 `HIAI_PATH=${OHOS_ARCH}`），
   需在 `source/backend/hiai/3rdParty/<ABI>/` 放置 `libhiai.so`、`libhiai_ir.so`、`libhiai_ir_build.so`
   与 `3rdParty/include/` 头文件（仓库 ENV.md 给出了依赖获取路径：CANN 与 command-line-tools 位于
   `/temp/csm/` 下）；
2. **运行路径①/②**：on-device 在线构图 —— 首帧 onResize 编译（大图可达分钟级），随后执行；
   生产环境建议开启路径①′（OM 缓存）：`MNN_HIAI_CACHE_OM_BY_CHUNK`，首帧后秒级启动（见 §1.1）；
3. **离线 OM（路径③，`.om` 由 OMG 工具生成，见 §1.2）**：
   - 用 `HIAI_DEBUG` 构建在设备上跑一次，`model.Save()` 序列化出 `test.irpb`；
   - 借助 DDK/CANN 的 command-line-tools 把 irpb 离线编译成 `.om`；
   - 应用层实现 `INpuChunkExecutor`，用 `HIAIModelManager`（OM）直接加载执行，
     MNN 侧配置 `visual_blocks_om_paths` 即可跳过在线编译（省启动时间与峰值内存）；
   - 也可直接使用 `MNN_HIAI_CACHE_OM_BY_CHUNK` 的在线缓存：首次编译后
     OM 自动缓存到 `npu_model_dir/chunk_<i>/vision.om`，下次启动直接加载。

---

## 8. 关键本地增强（区别于上游 MNN）

本仓库的 hiai 后端在上游基础上加入了大量鸿蒙落地修复（均以编译宏控制），排查问题时值得注意：

| 宏/机制 | 位置 | 作用 |
|---|---|---|
| `MNN_HIAI_USE_LOCAL_NPU_FIXES` | `NPUBackend.hpp:12` | 输入槽位按字节大小重建 mInputMap；输出按 MemObj 匹配（兼容 Tensor::clone）；5D 输入挤压；null 源保护 |
| `MNN_HIAI_FREE_CONST_HOST` | `NPUBackend.hpp:32` | 常量拷入 ge Const 后释放 host 内存，降峰值 RAM |
| `MNN_HIAI_CACHE_OM_BY_CHUNK` | CMake 选项 | chunk 级 OM 缓存到 `chunk_i/vision.om` |
| `HIAI_VERBOSE` / `HIAI_DEBUG` | CMake 选项 | 构图各阶段日志（定位 (a)~ (f) 六类失败）；debug 时 dump irpb/om |
| `HiAIDelegateBackend` + 70 上限 | `HiAIDelegateBackend.cpp:42` | per-conv 委托与 NPU 驱动资源保护 |
| `INpuChunkExecutor` | `npu_chunk_executor.hpp` | 应用层 OM 直通接口 |
| `pNPUModelDirPath` / `EXTERNAL_NPU_FILE_DIR` | `Backend.hpp:278` / `Interpreter.hpp:281` | 外部 NPU 缓存目录贯通 RuntimeManager → Backend |

---

## 9. 一句话总结

> MNN 在鸿蒙上通过 HiAI DDK 三库（ModelManager / IR / IR Builder）与昇腾 NPU 通信：
> **MNN 图 → 每个 op 的 Execution 在 onResize 时手工构建 `ge::Operator` 子图（mGrapMap 按 tensorIndex
> 连边）→ 汇总成 `ge::Graph/ge::Model` → `domi::HiaiIrBuild::BuildIRModel` 在线编译成 OM 二进制 →
> `AiModelMngerClient::Load` 加载进 NPU → 推理时通过 host 可见的 AiTensor（ION 内存）memcpy 交换数据**。
> 不支持 NPU 的 op 由调度层自动落到 CPU；LLM 场景下视觉编码器按 chunk 拆分、逐 chunk 在 NPU/CPU 之间
> 用 host 浮点数组桥接串行执行；预编译 `.om` 则由应用层通过 `INpuChunkExecutor` 直通，MNN 只做编排。
> 关于量化权重：8-bit（per-group）权重在**编图期由宿主 CPU 一次性反量化为 fp32** 烘焙进 IR，OM 权重段
> 默认 FP32 存储（4B/权重）、CUBE 以 fp16 MAC 计算，推理期零反量化 —— 详见 §3.6。
> 关于模型的来处：从 HF 定义到 ONNX、再到 MNN 融合 `OpType_Attention`、再到 ge 图展开的完整转换链路，
> 以 attention 为贯穿示例的逐步详解 —— 详见 §3.7。
> 关于离线图编译：`offline_hiai_npu` 分支实现了 PC 端 OMG 直编 `.om`（含量化）的完整闭环，
> 以及输出顺序 / OH API vs mnn::hiai / OMG-OM-OMC 等澄清 —— 详见 §1.3、§1.4。

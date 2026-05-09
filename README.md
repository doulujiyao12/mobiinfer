# Qwen3-VL 在高通 NPU / 麒麟 NPU 上运行说明

本文档说明如何在 mobiinfer 中准备并运行 Qwen3-VL。

- 第 1 部分：高通 NPU（支持离线交叉编译，分 chunk 主干text 网络 + fix-shape visual 网络 图片输入样例\<img\>test.jpg\<hw\>600,270\</hw\>\</img\>） 如果要改变图片输入尺寸，在下面docker_qnn编译阶段改变输入张量尺寸，现在 visual blocks 输入seqlen = 608 对应height = 600, weight = 270
- 第 2 部分：麒麟 NPU（fix-shape visual 网络）

## Demo（手机 GUI Agent 功能展示）

[![demo](doc/demo.gif)](doc/demo.mp4)

点击上图可打开/播放原始视频：[doc/demo.mp4](doc/demo.mp4)

以上demo仓库可以详见[mobiinfra-oh](https://github.com/doulujiyao12/mobiinfra-oh)

## 0. 量化与校准工具（mobi-autoround）

- 项目地址：<https://github.com/doulujiyao12/mobi-autoround>
- 该仓库支持自定义图片校准数据集，并导出 GPTQ 格式量化结果；导出的 GPTQ 格式可通过本仓库的 `llmexport` 进一步转换成 MNN 推理所需的文件格式。

---

## 1. 高通 NPU（Qualcomm）

### 1.0 Qualcomm NPU 依赖获取

如果你要在宿主机或开发机上准备 Qualcomm NPU 相关环境，可以按下面步骤获取依赖：

1. 注册高通账号：<https://myaccount.qualcomm.com/signup>
2. 访问 Qualcomm AI Engine Direct SDK（QNN SDK），下载 SDK 并解压到本地目录。
  - 示例路径：`/home/xiaying/third/qnn/qairt/2.38.0.250901`
3. 修改 `~/.bashrc`，把 SDK 路径加入环境变量，然后执行 `source ~/.bashrc`，或者重新打开终端。

示例配置：

```bash
export QNN_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export QNN_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export HEXAGON_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
```

### 1.1 在 Ubuntu x86 上编译 QNN 交叉编译中间工具

> 这一步用于产出 QNN SDK 交叉编译流程需要的中间工具，**不会**产出可直接在手机侧运行的二进制文件。

```bash
cd ./mobiinfer
mkdir build_qnn_x86
cd build_qnn_x86
cmake .. \
  -DMNN_BUILD_LLM=true \
  -DMNN_LOW_MEMORY=true \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_BUILD_TEST=ON \
  -DMNN_QNN=ON \
  -DMNN_QNN_CONVERT_MODE=ON \
  -DMNN_WITH_PLUGIN=OFF \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON

make -j64
```

### 1.2 编译 Android 侧可执行文件（llm_demo）

> 这一步用于编译可在高通手机上运行的可执行文件，例如 `llm_demo`。

```bash
cd ./project/android
mkdir build
cd build
../build_64.sh \
  -DMNN_SUPPORT_BF16=true \
  -DMNN_BUILD_LLM=true \
  -DMNN_ARM82=true \
  -DMNN_OPENCL=true \
  -DMNN_USE_LOGCAT=true \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_LOW_MEMORY=true \
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
  -DMNN_IMGCODECS=true \
  -DMNN_QNN=ON \
  -DMNN_WITH_PLUGIN=ON \
  -DMNN_QNN_CONVERT_MODE=OFF
```

### 1.3 量化模型准备

如果采用 `mobi-autoround` 产出的 GPTQ 量化结果，请在导出时添加 `--gptq_path` 指向对应的 GPTQ 模型目录：

```bash
cd ./transformers/llm/export
python llmexport.py --path /origin/fp/model/path \
    --export mnn --gptq_path /gptq/model/path --quant_bit 4 --quant_block 128 \
    --visual_quant_bit 4 --visual_quant_block 128 --lm_quant_bit 16 \
    --seperate_embed --visual_split
```

### 1.4 构建 qnn_docker（生成 QNN 模型离线转换的 bin 权重与执行图）

> 该步骤用于构建 `qnn_docker` 环境，离线生成 QNN 模型转换所需的 `bin` 权重文件与执行图。

- 参考文档：[qnn_docker/README.md](qnn_docker/README.md)

### 1.5 推送 QNN 运行时依赖与模型并执行（Android）

首先将生成的./project/android/build 中的 llm_demo文件和so文件（包括tools/cv/libMNNOpenCV.so 和audio/libMNNAudio.so，中间编译产出不需要）推送到手机的指定目录 (PHONEDIR = /data/local/tmp/mobiinfer 可以指定任何可执行权限的目录下)：

将 QNN 相关运行时库推送到 Android 侧测试目录：

```bash
ANDROID_WORKING_DIR=/data/local/tmp/mobiinfer/qnn_sdk
HEXAGON_ARCH=75
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${HEXAGON_ARCH}Stub.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v${HEXAGON_ARCH}/unsigned/libQnnHtpV${HEXAGON_ARCH}Skel.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${ANDROID_WORKING_DIR}
```

推送模型：

```bash

cd transformers/llm/export
adb push model /data/local/tmp/mobiinfer/model
```

手机上运行：

```bash
export ADSP_LIBRARY_PATH=/qnn/sdk:$ADSP_LIBRARY_PATH
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:{ANDROID_WORKING_DIR}:{PHONEDIR}:$LD_LIBRARY_PATH

cd ${PHONEDIR}
./llm_demo model/config_qnn.json
```

一个典型的 `config_qnn.json` 示例：

```json
{
  "llm_model": "qnn/llm.mnn",
  "chunk_limits": [128, 1],
  "backend_type": "cpu",
  "thread_num": 4,
  "precision": "low",
  "memory": "low",
  "sampler_type": "mixed",
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "min_p": 0.05,
  "tfs_z": 1.0,
  "typical": 0.95,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "penalty_window": 0,
  "n_gram": 8,
  "ngram_factor": 1.0,
  "tokenizer_file": "tokenizer.mtok",
  "mllm": {
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "normal",
    "memory": "low"
  },
  "visual_split": true,
  "visual_pre_model": "visual_pre.mnn",
  "visual_blocks_model": "visual_blocks_69_79.mnn",
  "visual_post_model": "visual_post.mnn",
  "visual_blocks_backend_type": "npu"
}
```

其中：

- `qnn/llm.mnn` 是主干 text 网络转化后的 QNN bin 和 MNN 文件（同名 `.mnn` 对应 QNN 的权重与执行图产物）。
- `visual_blocks_69_79.mnn` 是图片 visual 网络（blocks）转化后的 QNN bin 和 MNN 文件。

### 1.6 结果说明

- `build_qnn_x86` 阶段：提供 QNN 相关中间工具（用于转换/交叉编译流程）
- `project/android/build` 阶段：产出手机侧可运行程序（含 `llm_demo`）

---

## 2. 麒麟 NPU（Kirin）

下面给出在本仓库中准备并在麒麟 NPU（HiAI/Huawei）上构建运行的建议流程与示例命令。

### 2.1 下载并准备 CANN Kit

1. 从华为开发者网站下载 CANN-Kit-next-6.0.1.0：

  https://developer.huawei.com/consumer/cn/doc/hiai-Library/ddk-download-0000001053590180

2. 解压后，将其中的 `arm64-v8a` 与 `include` 两个目录拷贝到仓库的第三方路径：

```bash
# 假设已将包解压到 ~/downloads/CANN-Kit-next-6.0.1.0
cp -r ~/downloads/CANN-Kit-next-6.0.1.0/ddk/ai_ddk_lib/lib64/* ./source/backend/hiai/3rdParty/arm64-v8a
cp -r ~/downloads/CANN-Kit-next-6.0.1.0/ddk/ai_ddk_lib/include/* ./source/backend/hiai/3rdParty/include
```

（目标位置：`source/backend/hiai/3rdParty/arm64-v8a` 和 `source/backend/hiai/3rdParty/include`）

### 2.2 下载 Huawei Command Line Tools

1. 从华为开发者官网下载 Command Line Tools（用于 HarmonyOS/鸿蒙 构建工具链）：

  https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos?ha_source=sousuo&ha_sourceId=89000251

2. 解压或放置到合适位置，并设置环境变量 `HARMONY_HOME` 指向解压后的 OpenHarmony SDK 路径，例如：

```bash
# 假设解压后 sdk 在 commandline-tools/command-line-tools/sdk/default/openharmony/
export HARMONY_HOME=/path/to/commandline-tools/command-line-tools/sdk/default/openharmony/
```

（请根据实际解压路径替换 `/path/to/...`）

### 2.3 导出适配 Kirin NPU 的 MNN 模型

如果采用 `mobi-autoround` 产出的 GPTQ 量化结果，请在导出时添加 `--gptq_path` 指向对应的 GPTQ 模型目录：

在导出 Qwen3-VL 的 MNN 模型时，可以通过下面命令对视觉分支进行切分，降低 Kirin NPU 在线编图时的内存压力：

```bash
cd ./transformers/llm/export
python llmexport.py --path /origin_fp/model_path \
    --export mnn --gptq_path /gptq/model/path --quant_bit 4 --quant_block 128 \
    --visual_quant_bit 4 --visual_quant_block 128 --lm_quant_bit 16 \
    --seperate_embed --visual_split --visual_npu_chunk 6 \
    --visual_chunk_backends "npu,npu,npu,npu,cpu,cpu"
```

参数说明：

- `--visual_npu_chunk 6` 表示把视觉头切分成 6 份。Kirin NPU 在线编译时，如果单个 graph 过大，容易报 `Low memory` 错误，因此将视觉部分拆成 6 份来减少单次编图压力。
- `--visual_chunk_backends "npu,npu,npu,npu,cpu,cpu"` 表示这 6 份分别使用哪些后端执行。上面的配置表示前 4 份跑在 NPU，后 2 份跑在 CPU。
- 不建议 6 份全部都配置为 `npu`，否则在部分机型或大模型场景下，应用可能会直接 crash。

### 2.4 编译仓库中的 Harmony/鸿蒙 端库（生成 `libMNN.so`）

进入构建目录并运行仓内提供的构建脚本：

```bash
cd ./project/harmony
mkdir -p build
cd build
../build_64.sh
```

运行成功后，会在相应输出目录生成 `libMNN.so`（或位于 `build/output` / `build/lib` 等子目录，视 `build_64.sh` 脚本实现而定）。

### 2.5 使用鸿蒙 App 进行测试

由于鸿蒙系统不支持命令行开发，我们开发了鸿蒙 App 进行测试。编译得到的 `libMNN.so` 需要替换到 [mobiinfra-oh](https://github.com/doulujiyao12/mobiinfra-oh) 仓库中对应位置：

- https://github.com/doulujiyao12/mobiinfra-oh/blob/dev/entry/libs/arm64-v8a/libMNN.so

### 2.6 说明与注意事项

- 请确保 `source/backend/hiai/3rdParty/arm64-v8a` 和 `.../include` 已存在且内容完整。缺少头文件或库会导致编译失败。
- `HARMONY_HOME` 必须指向命令行工具提供的 OpenHarmony SDK 根目录，否则构建脚本找不到工具链。
- 若构建失败，请查阅 `project/harmony/build_64.sh` 中的日志与输出路径，按错误提示补充依赖。
- 本节假定你已经在机器上安装并配置好对应的交叉编译工具链以及必要的 Android/Harmony 环境变量。

---

---

## 致谢

- 感谢 [alibaba/MNN](https://github.com/alibaba/MNN) 开源仓库提供的基础能力与工程实现。

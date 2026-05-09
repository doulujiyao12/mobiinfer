# mnn-qnn Docker 使用说明

## 概览

- 本镜像 `mobiinfer-qnn:22.04` 用于在 x86_64 上运行 MNN + Qualcomm QNN SDK 相关工具。
- 约定路径：原始 QNN SDK 挂载到 `/opt/qnn`（只读），MNN 源代码挂载到 `/workspace/mobiinfer`，QNN mirror 放在 `/root/qnn-mirror`。

## 快速开始

### 1. 构建镜像

在包含 `Dockerfile.mnn_qnn` 的目录下执行：

```bash
docker build -t mobiinfer-qnn:22.04 -f Dockerfile.mnn_qnn .
```

<!-- ### 2. 交互式运行容器(不需要)

```bash
docker run --rm -it --name mobiinfer-qnn \
  -v /path/to/mobiinfer:/workspace/mobiinfer \
  -v /path/to/qnn-sdk:/opt/qnn:ro \
  -e BUILD_QNN_MIRROR=0 \
  mobiinfer-qnn:22.04 /bin/bash -->
```

### 2. 启动时自动构建 QNN mirror

镜像启动后会把 `/opt/qnn` 复制到 `/root/qnn-mirror`，并对 QNN 的 `Makefile` 应用单行 patch：

```bash
docker run --rm -it --name mobiinfer-qnn-build \
  -v /path/to/mobiinfer:/workspace/mobiinfer \
  -v /path/to/qnn-sdk:/opt/qnn:ro \
  -e BUILD_QNN_MIRROR=1 \
  mobiinfer-qnn:22.04 /bin/bash
```

注意：第一次构建 mirror 可能较慢，也会额外占用磁盘空间。

## 在容器内运行 QNN 生成脚本

在容器内先升级 `libc++`，再设置环境变量，然后执行 QNN 生成流程。

### 第一步：升级 libc++

```bash
sudo apt update
sudo apt install -y libc++1 libc++-dev
```

### 第二步：设置 `LD_LIBRARY_PATH`

```bash
export LD_LIBRARY_PATH=/workspace/mobiinfer/build_qnn_x86:$LD_LIBRARY_PATH
```

<!-- 如果还需要使用 mirror 的 QNN 库，可再追加：

```bash
export LD_LIBRARY_PATH=/root/qnn-mirror/lib/x86_64-linux-clang:$LD_LIBRARY_PATH -->
```

### 第三步：生成 qwen3vl 的 text 主干 QNN bin

```bash
cd /workspace/mobiinfer/transformers/llm/export/npu
python3 generate_llm_qnn.py \
  --model ../model \
  --target llm --chunk_size 128 --max_history_token 2048 \
  --soc_id 69 --dsp_arch v79 \
  --mnn_path /workspace/mobiinfer/build_qnn_x86
```

这一步会生成 qwen3vl 中 text 主干网络的分 chunk QNN bin 文件。

### 第四步：生成固定图片 shape 的 QNN bin

```bash
cd /workspace/mobiinfer/build_qnn_x86
./MNN2QNNModel /opt/qnn 69 79 \
  /workspace/mobiinfer/transformers/llm/export/model/visual_blocks.mnn \
  /workspace/mobiinfer/transformers/llm/export/model/qnn_visual \
  1 608x1024_2x1x608x1x64_1x608x60
```

这一步用于生成固定图片 shape 的 QNN bin。

## 关于 `qnn-shim`（clang++ -> g++）

- 镜像里提供了 `qnn-shim`，把 `/root/bin/qnn-shim` 放到 `PATH` 前面。
- 其中 `clang++` 实际会转发到 `/usr/bin/g++-11`，这样可以兼容 QNN 工具链对 `clang++` 的调用。

容器内可验证：

```bash
echo "$QNN_SDK_ROOT"
which clang++
```

## Mirror 的 Makefile patch

为避免权重被放进 `.data` 后出现 `R_X86_64_PC32` 或 PC32 truncation 问题，mirror 脚本会把：

```make
TARGET_OBJCOPY_CMD := objcopy -I binary -O elf64-x86-64 -B i386:x86-64
```

替换为：

```make
TARGET_OBJCOPY_CMD := objcopy -I binary -O elf64-x86-64 -B i386:x86-64 --rename-section .data=.ldata,alloc,load,readonly,data,contents
```

这样权重会放入 `.ldata` 节，减少链接阶段出错的概率。

## 常见问题

### 1. 容器启动报 `exec format error`

通常是 entrypoint 或 shim 文件在镜像内为空（0 字节），或者没有执行权限。建议检查：

```bash
ls -l /usr/local/bin/mnn-qnn-entrypoint.sh
ls -l /root/bin/qnn-shim/clang++
file /usr/local/bin/mnn-qnn-entrypoint.sh
```

### 2. 删除同名容器

```bash
docker rm -f mnn-qnn || true
```

### 3. 重新进入正在运行的容器

```bash
docker exec -it mnn-qnn /bin/bash
```

### 4. 查看容器日志

```bash
docker logs <container-id-or-name>
```

### 5. 查看所有容器

```bash
docker ps -a
```

## 建议

- 不建议把完整 QNN SDK 直接烧进镜像，体积和许可都不太合适；推荐运行时通过 `-v /path/to/qnn-sdk:/opt/qnn:ro` 挂载。
- 如果你经常重复执行这些命令，我可以继续帮你补一个一键运行脚本 `run_mnn_qnn.sh`。

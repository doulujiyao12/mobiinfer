#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 一次性流程: 用 256 张真实图像生成真实校准输入, 再用真实校准输入
# 生成全部 6 个 visual chunk 的量化 OM 文件。
#
# 链路:
#   真实图像(256张, 强制 hw=600,270 -> seq_len=608)
#     -> MNN 引擎 (MNN_VISUAL_CHUNK_INPUT_DUMP=ON, chunk_backends=cpu)
#     -> dump raw float32 bin + meta json   (/temp/fdh/input_calib/chunk_dump_raw/)
#     -> bin_to_chunk_npz.py 转 fp16 npz    (/temp/fdh/input_calib/calib_inputs_256/)
#     -> visual_plugin_quant_matmul_route.py (每 chunk, --num_samples 256)
#     -> run_visual_plugin_matmul_omc.sh    (/temp/fdh/model_omc/model_..._chunk{i}_real256/)
#     -> 6 个 visual_plugin_matmul_quantized.om
#
# 中间产物全部放 /temp/fdh/input_calib/ (本地空间有限, 故用 /temp).
# 流程结束后可选清理 raw bin (KEEP_RAW=0).
# ============================================================

# ============================== 可调参数 ==============================
NUM_IMAGES=256                  # dump 几张图 (同时 = 量化用几个样本)
HW_OVERRIDE="600,270"           # 强制缩放尺寸 -> seq_len=608
SEED=42                         # 采样随机种子 (可复现)

CALIB_WORK_ROOT=/temp/fdh/input_calib_fdh            # 所有中间产物根目录
SRC_IMG_DIR=/temp/csm/sft-0422-quant-500-half-size  # 496 张真实图
MODEL_CFG_DIR=/temp/fdh/baiducloud/902137265_doulujiyao1/model_6chunk_nor_kirinnpu_visual4

PROMPT_FILE=${CALIB_WORK_ROOT}/image_prompt.txt
# config 必须生成在模型目录里: config 用相对路径引用 llm.mnn/tokenizer.mtok 等,
# llm_demo 按所在目录解析这些路径. 放到 CALIB_WORK_ROOT 会导致找不到模型文件.
CONFIG_DUMP=${MODEL_CFG_DIR}/config_dump_608.json
DUMP_RAW_DIR=${CALIB_WORK_ROOT}/chunk_dump_raw
CALIB_NPZ_DIR=${CALIB_WORK_ROOT}/calib_inputs_256

OMC_OUT_ROOT=/temp/fdh/model_omc_fdh
ROUTE_SUFFIX=real256            # route_dir 后缀
NPU_CHUNKS=6
PLATFORM=kirin9020

KEEP_RAW=0                     # 1=保留 raw bin, 0=转完 npz 后清理 (省空间)
SKIP_DUMP=0                     # 1=跳过 dump (复用已有 DUMP_RAW_DIR)
SKIP_NPZ=0                      # 1=跳过 bin->npz (复用已有 CALIB_NPZ_DIR)
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # mobiinfer
LLM_DEMO_BIN=${REPO_ROOT}/build_x86/llm_demo

set -e

echo "=========================================="
echo "  真实图像校准量化全流程"
echo "  num_images   : ${NUM_IMAGES}"
echo "  hw_override  : ${HW_OVERRIDE}  (seq_len=608)"
echo "  work_root    : ${CALIB_WORK_ROOT}"
echo "  skip_dump    : ${SKIP_DUMP}"
echo "  skip_npz     : ${SKIP_NPZ}"
echo "=========================================="
echo ""

mkdir -p "${CALIB_WORK_ROOT}" "${OMC_OUT_ROOT}"

# ---------- 0) 前置检查 ----------
if [ ! -x "${LLM_DEMO_BIN}" ]; then
    echo "ERROR: llm_demo 不存在或不可执行: ${LLM_DEMO_BIN}"
    echo "       请先按 README '步骤1' 编译 (开启 -DMNN_VISUAL_CHUNK_INPUT_DUMP=ON)"
    exit 1
fi
if [ ! -d "${MODEL_CFG_DIR}" ]; then
    echo "ERROR: 6chunk 模型目录不存在: ${MODEL_CFG_DIR}"
    exit 1
fi
# 检查 dump 宏是否真的开在 build_x86 里 (CMakeCache)
if ! grep -q "MNN_VISUAL_CHUNK_INPUT_DUMP:BOOL=ON" "${REPO_ROOT}/build_x86/CMakeCache.txt" 2>/dev/null; then
    echo "ERROR: build_x86 未开启 MNN_VISUAL_CHUNK_INPUT_DUMP, dump 不会触发"
    echo "       cd build_x86 && cmake .. -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON \\"
    echo "         -DMNN_VISUAL_CHUNK_INPUT_DUMP=ON && make -j\$(nproc) llm_demo"
    exit 1
fi
echo "[check] llm_demo + dump宏 + 模型目录  OK"
echo ""

# ---------- 1) 采样图片, 生成 image_prompt.txt ----------
echo "[step 1] 采样 ${NUM_IMAGES} 张图 -> ${PROMPT_FILE}"
python3 "${SCRIPT_DIR}/select_images.py" \
    --src_dir "${SRC_IMG_DIR}" \
    --out_prompt "${PROMPT_FILE}" \
    --num "${NUM_IMAGES}" \
    --hw "${HW_OVERRIDE}" \
    --seed "${SEED}"
echo ""

# ---------- 2) 生成 config_dump.json (chunk_backends 全 cpu + dump 配置) ----------
echo "[step 2] 生成 dump 配置 -> ${CONFIG_DUMP}"
python3 - "${MODEL_CFG_DIR}/config.json" "${CONFIG_DUMP}" "${DUMP_RAW_DIR}" "${NUM_IMAGES}" <<'PY'
import json, sys
src, dst, dump_dir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = json.load(open(src, 'r', encoding='utf-8'))
# host 上 hiai NPU 不可用, 必须全置 cpu 才能走 MNN-module 路径触发 dump
cfg["visual_blocks_chunk_backends"] = ["cpu"] * len(cfg["visual_blocks_chunks"])
# dump 配置
cfg["visual_chunk_input_dump_dir"] = dump_dir
cfg["visual_chunk_input_dump_samples"] = n
json.dump(cfg, open(dst, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(f"  chunk_backends -> all cpu ({len(cfg['visual_blocks_chunks'])} chunks)")
print(f"  dump_dir      -> {dump_dir}")
print(f"  dump_samples  -> {n}")
PY
echo ""

# ---------- 3) 跑 llm_demo, dump 出 raw bin ----------
if [ "${SKIP_DUMP}" = "1" ]; then
    echo "[step 3] SKIP_DUMP=1, 复用已有 dump: ${DUMP_RAW_DIR}"
else
    echo "[step 3] 跑 llm_demo, dump raw bin -> ${DUMP_RAW_DIR}"
    rm -rf "${DUMP_RAW_DIR}"
    mkdir -p "${DUMP_RAW_DIR}"
    # 注意: 必须一次 llm_demo 进程跑完全部图 (dump 计数器进程内从0计)
    # max_token=1 只要视觉前向, 但 llm_demo 默认 max_token=-1 会继续生成,
    # 为节省时间强制只生成 1 token.
    cd "${REPO_ROOT}/build_x86"
    if ! ./llm_demo "${CONFIG_DUMP}" "${PROMPT_FILE}" 1; then
        echo "ERROR: llm_demo 运行失败"
        exit 1
    fi
    cd "${SCRIPT_DIR}"
    # 期望文件数: 6 chunk * N sample * (3 tensor + 1 meta) = 24*N
    EXPECT=$((24 * NUM_IMAGES))
    GOT=$(ls "${DUMP_RAW_DIR}" 2>/dev/null | wc -l)
    echo "  dump 文件数: ${GOT} / 期望 ${EXPECT}"
    if [ "${GOT}" -ne "${EXPECT}" ]; then
        echo "WARN: dump 文件数不等于期望, 可能部分图未触发 dump (检查日志)"
    fi
fi
echo ""

# ---------- 4) bin + json -> npz ----------
if [ "${SKIP_NPZ}" = "1" ]; then
    echo "[step 4] SKIP_NPZ=1, 复用已有 npz: ${CALIB_NPZ_DIR}"
else
    echo "[step 4] bin -> npz -> ${CALIB_NPZ_DIR}"
    rm -rf "${CALIB_NPZ_DIR}"
    python3 "${SCRIPT_DIR}/bin_to_chunk_npz.py" \
        --dump_dir "${DUMP_RAW_DIR}" \
        --out_dir "${CALIB_NPZ_DIR}" \
        --dtype fp16
    # 校验格式 (与现有 seq608 随机版逐 key 对比)
    REF_DIR="${SCRIPT_DIR}/model_visual_plugin_matmul_calib_seq608/calib_inputs"
    if [ -d "${REF_DIR}" ]; then
        python3 - "${CALIB_NPZ_DIR}" "${REF_DIR}" <<'PY'
import numpy as np, sys, glob, os
calib, ref = sys.argv[1], sys.argv[2]
keys=["hidden_states_in","rotary_pos_emb","attention_mask"]
real = sorted(glob.glob(os.path.join(calib, "chunk_00_sample_*.npz")))
if not real:
    print("  WARN: 无 chunk_00 npz, 跳过格式校验"); raise SystemExit(0)
r = np.load(real[0])
ref_f = os.path.join(ref, "chunk_00_sample_000.npz")
if not os.path.exists(ref_f):
    print("  WARN: 无 ref, 跳过格式校验"); raise SystemExit(0)
rr = np.load(ref_f)
ok = all(r[k].shape==rr[k].shape and r[k].dtype==rr[k].dtype for k in keys)
print(f"  chunk_00 sample0: hidden={r['hidden_states_in'].shape} rotary={r['rotary_pos_emb'].shape} mask={r['attention_mask'].shape}")
print(f"  格式对比随机 seq608 版: {'OK' if ok else 'MISMATCH'}")
assert ok, "格式不匹配! 检查 bin_to_chunk_npz / hw override"
PY
    fi
fi
echo ""

# ---------- 4.5) 可选清理 raw bin ----------
if [ "${KEEP_RAW}" = "0" ] && [ "${SKIP_DUMP}" != "1" ]; then
    echo "[step 4.5] 清理 raw bin (KEEP_RAW=0): ${DUMP_RAW_DIR}"
    rm -rf "${DUMP_RAW_DIR}"
fi
echo ""

# ---------- 5) 6 chunk 量化 + 导出 ONNX + OMC ----------
echo "[step 5] 逐 chunk 跑 量化+导出ONNX+OMC (num_samples=${NUM_IMAGES})"
echo ""

# ---- 环境 ----
source /opt/conda/etc/profile.d/conda.sh
conda activate CANN
export DDK_DOPT=/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=${DDK_DOPT}:${REPO_ROOT}/transformers/llm/export:${PYTHONPATH:-}

cd "${SCRIPT_DIR}"

FAILED_CHUNKS=()
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W8A8_${i}_${ROUTE_SUFFIX}"
    ONNX_FILE="${ROUTE_DIR}/onnx/visual_blocks_npu_${i}.onnx"
    OM_FILE="${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized.om"
    OMC_LOG="${ROUTE_DIR}/omc_output/omc_${PLATFORM}.log"

    echo "=========================================="
    echo "  chunk ${i} / 5"
    echo "  route: ${ROUTE_DIR}"
    echo "=========================================="

    if [ -f "${OM_FILE}" ]; then
        echo "[chunk ${i}] OM 已存在, 跳过: ${OM_FILE}"
        ls -lh "${OM_FILE}" 2>/dev/null | awk '{print "  size: "$5}'
        echo ""
        continue
    fi

    echo "[chunk ${i}] running prepare + calibrate + export-onnx (num_samples=${NUM_IMAGES}) ..."
    if ! python visual_plugin_quant_matmul_route.py \
        --route_dir "${ROUTE_DIR}" \
        --chunk_index ${i} \
        --npu_chunks ${NPU_CHUNKS} \
        --quant_strategy Quant_aigc_ptq \
        --weight_bit 8 \
        --weight_algo min_max \
        --act_bit 8 \
        --input_algo min_max \
        --num_samples ${NUM_IMAGES} \
        --group_size 128 \
        --use_qwen3_style_rotary \
        --input_dir "${CALIB_NPZ_DIR}" \
        --force_regen \
        all; then
        echo "[chunk ${i}] ERROR: all step 失败"
        FAILED_CHUNKS+=("chunk_${i}:all")
        echo ""
        continue
    fi
    echo "[chunk ${i}] ONNX 已导出: ${ONNX_FILE}"

    echo "[chunk ${i}] running OMG (${PLATFORM}) ..."
    if PLATFORM="${PLATFORM}" bash run_visual_plugin_matmul_omc.sh \
            "${ROUTE_DIR}" fp16 > "${OMC_LOG}" 2>&1; then
        if [ -f "${OM_FILE}" ] && grep -q "OMG generate offline model success" "${OMC_LOG}"; then
            echo "[chunk ${i}] OM SUCCESS: ${OM_FILE}"
            ls -lh "${OM_FILE}" 2>/dev/null | awk '{print "  size: "$5}'
        else
            echo "[chunk ${i}] OMG 跑完但 OM 缺失或无成功标记, 查日志: ${OMC_LOG}"
            FAILED_CHUNKS+=("chunk_${i}:omc_check")
        fi
    else
        echo "[chunk ${i}] OMG 失败, 查日志: ${OMC_LOG}"
        FAILED_CHUNKS+=("chunk_${i}:omc")
    fi
    echo ""
done

# ---------- 6) 汇总 ----------
echo "=========================================="
echo "  pipeline finished"
echo "=========================================="
ALL_OK=1
for i in 0 1 2 3 4 5; do
    OM_FILE="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W8A8_${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.om"
    if [ -f "${OM_FILE}" ]; then
        SIZE=$(ls -lh "${OM_FILE}" 2>/dev/null | awk '{print $5}')
        echo "  [OK]   chunk ${i}: ${OM_FILE}  (${SIZE})"
    else
        echo "  [FAIL] chunk ${i}: 缺失 ${OM_FILE}"
        ALL_OK=0
    fi
done

echo ""
echo "中间产物: ${CALIB_WORK_ROOT}/"
echo "  image_prompt.txt   ($(wc -l < "${PROMPT_FILE}" 2>/dev/null) 行)"
echo "  config_dump_608.json"
[ -d "${DUMP_RAW_DIR}" ] && echo "  chunk_dump_raw/   (保留)"
echo "  calib_inputs_256/ ($(ls "${CALIB_NPZ_DIR}"/*.npz 2>/dev/null | wc -l) npz)"

if [ ${#FAILED_CHUNKS[@]} -gt 0 ]; then
    echo ""
    echo "Failures: ${FAILED_CHUNKS[*]}"
    exit 1
fi
if [ ${ALL_OK} -ne 1 ]; then
    echo "部分 OM 缺失。"
    exit 1
fi
echo ""
echo "全部 6 个 chunk 的 OM 已用真实校准输入 (256 张图, seq_len=608) 生成完毕。"
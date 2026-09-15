#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 快速验证: 只用 1 个校准样本 (num_samples=1) 跑 W8A8 量化,
# 用 complete DDK 6.1.1.0-complete 编译成 OMC (--target=omc),
# 真正把量化落到 NPU 量化算子 (QuantBatchMatmulV3 等)。
#
# 与 run_real_calib_1_W8A8.sh / _v6110.sh 的区别:
#   - 输出 .omc (不是 .om) -> 真 NPU 量化
#   - 用 complete 环境 + 显式 omc 编译环境变量
#   - 调用 run_visual_plugin_matmul_omc_v6110_omc.sh
#
# 校准输入目录复用 /temp/fdh/input_calib_fdh/calib_inputs_16
#   - 每个 chunk 有 sample_000/016/.../240 共 16 个 npz
#   - load_calibration_samples 会取排序后的前 1 个 (sample_000)
#
# 链路 (仅量化部分):
#   calib_inputs_16 (fp16 npz, 取 1 个)
#     -> visual_plugin_quant_matmul_route.py (每 chunk, --num_samples 1, W8A8)
#     -> run_visual_plugin_matmul_omc_v6110_omc.sh (/temp/fdh/model_omc_fdh_9030/model_..._chunk_W8A8_{i}_real1_w8a8_omc/)
#     -> 6 个 visual_plugin_matmul_quantized.omc
#     -> 汇总重命名为 visual_blocks_npu_{0..5}.omc
# ============================================================

# ============================== 可调参数 ==============================
NUM_SAMPLES=1                   # 快速验证: 只用 1 个校准样本
NPU_CHUNKS=6
PLATFORM=kirin9030

# 复用已生成的 16 样本校准 npz (只取前 1 个做校准)
CALIB_NPZ_DIR=/temp/fdh/input_calib_fdh/calib_inputs_16

OMC_OUT_ROOT=/temp/fdh/model_omc_fdh_9030
ROUTE_SUFFIX=real1_w8a8_omc      # route_dir 后缀 (独立)
OM_COLLECT_DIR=${OMC_OUT_ROOT}/model_visual_plugin_matmul_W8A8_${ROUTE_SUFFIX}   # 汇总目录
OMC_SCRIPT=run_visual_plugin_matmul_omc_v6110_omc.sh
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # mobiinfer

set -e

echo "=========================================="
echo "  快速验证量化 (W8A8 -> OMC, 1 样本, DDK 6.1.1.0-complete)"
echo "  num_samples  : ${NUM_SAMPLES}"
echo "  act_bit      : 8  (weight_bit: 8)"
echo "  target       : omc (真 NPU 量化)"
echo "  platform     : ${PLATFORM}"
echo "  calib_npz    : ${CALIB_NPZ_DIR}"
echo "  route_suffix : ${ROUTE_SUFFIX}"
echo "=========================================="
echo ""

mkdir -p "${OMC_OUT_ROOT}"

# ---------- 0) 前置检查 ----------
if [ ! -d "${CALIB_NPZ_DIR}" ]; then
    echo "ERROR: 校准 npz 目录不存在: ${CALIB_NPZ_DIR}"
    exit 1
fi
CALIB_NPZ_COUNT=$(ls "${CALIB_NPZ_DIR}"/*.npz 2>/dev/null | wc -l)
echo "[check] 校准 npz 目录文件数: ${CALIB_NPZ_COUNT} (每个 chunk 取前 1 个)"
if [ ! -f "${SCRIPT_DIR}/${OMC_SCRIPT}" ]; then
    echo "ERROR: 缺少 OMC 编译脚本: ${SCRIPT_DIR}/${OMC_SCRIPT}"
    exit 1
fi
echo ""

# ---------- 5) 6 chunk 量化 + 导出 ONNX + OMC ----------
echo "[step 5] 逐 chunk 跑 量化+导出ONNX+OMC (W8A8 -> omc, num_samples=${NUM_SAMPLES})"
echo ""

# ---- 环境 ----
source /opt/conda/etc/profile.d/conda.sh
conda activate CANN
export DDK_DOPT=/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=${DDK_DOPT}:${REPO_ROOT}/transformers/llm/export:${PYTHONPATH:-}

cd "${SCRIPT_DIR}"

FAILED_CHUNKS=()
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W8A8_${i}_${ROUTE_SUFFIX}"
    ONNX_FILE="${ROUTE_DIR}/onnx/visual_blocks_npu_${i}.onnx"
    OMC_FILE="${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized.omc"
    OMC_LOG="${ROUTE_DIR}/omc_output/omc_${PLATFORM}.log"

    echo "=========================================="
    echo "  chunk ${i} / 5"
    echo "  route: ${ROUTE_DIR}"
    echo "=========================================="

    if [ -f "${OMC_FILE}" ]; then
        echo "[chunk ${i}] OMC 已存在, 跳过: ${OMC_FILE}"
        ls -lh "${OMC_FILE}" 2>/dev/null | awk '{print "  size: "$5}'
        echo ""
        continue
    fi

    echo "[chunk ${i}] running prepare + calibrate + export-onnx (W8A8, num_samples=${NUM_SAMPLES}) ..."
    if ! python visual_plugin_quant_matmul_route.py \
        --route_dir "${ROUTE_DIR}" \
        --chunk_index ${i} \
        --npu_chunks ${NPU_CHUNKS} \
        --quant_strategy Quant_aigc_ptq \
        --weight_bit 8 \
        --weight_algo min_max \
        --act_bit 8 \
        --input_algo min_max \
        --num_samples ${NUM_SAMPLES} \
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

    echo "[chunk ${i}] running OMG -> omc (${PLATFORM}) ..."
    if PLATFORM="${PLATFORM}" bash "${OMC_SCRIPT}" \
            "${ROUTE_DIR}" fp16 > "${OMC_LOG}" 2>&1; then
        if [ -f "${OMC_FILE}" ] && grep -q "OMG generate offline model success" "${OMC_LOG}"; then
            echo "[chunk ${i}] OMC SUCCESS: ${OMC_FILE}"
            ls -lh "${OMC_FILE}" 2>/dev/null | awk '{print "  size: "$5}'
        else
            echo "[chunk ${i}] OMG 跑完但 OMC 缺失或无成功标记, 查日志: ${OMC_LOG}"
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
    OMC_FILE="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W8A8_${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.omc"
    if [ -f "${OMC_FILE}" ]; then
        SIZE=$(ls -lh "${OMC_FILE}" 2>/dev/null | awk '{print $5}')
        echo "  [OK]   chunk ${i}: ${OMC_FILE}  (${SIZE})"
    else
        echo "  [FAIL] chunk ${i}: 缺失 ${OMC_FILE}"
        ALL_OK=0
    fi
done

echo ""
echo "校准输入: ${CALIB_NPZ_DIR}/ (每 chunk 取前 1 个, 共 6 个样本)"

if [ ${#FAILED_CHUNKS[@]} -gt 0 ]; then
    echo ""
    echo "Failures: ${FAILED_CHUNKS[*]}"
    exit 1
fi
if [ ${ALL_OK} -ne 1 ]; then
    echo "部分 OMC 缺失。"
    exit 1
fi

# ---------- 7) 汇总 OMC 到单一目录 ----------
echo ""
echo "[step 7] 汇总 OMC -> ${OM_COLLECT_DIR}"
mkdir -p "${OM_COLLECT_DIR}"
for i in 0 1 2 3 4 5; do
    SRC_OMC="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W8A8_${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.omc"
    DST_OMC="${OM_COLLECT_DIR}/visual_blocks_npu_${i}.omc"
    if [ -f "${SRC_OMC}" ]; then
        cp -f "${SRC_OMC}" "${DST_OMC}"
        echo "  [OK]   visual_blocks_npu_${i}.omc  ($(ls -lh "${DST_OMC}" 2>/dev/null | awk '{print $5}'))"
    else
        echo "  [FAIL] 缺失源 OMC: ${SRC_OMC}"
        ALL_OK=0
    fi
done
if [ ${ALL_OK} -ne 1 ]; then
    echo "汇总 OMC 缺失, 请检查。"
    exit 1
fi
echo ""
echo "汇总目录: ${OM_COLLECT_DIR}/"
ls -lh "${OM_COLLECT_DIR}"/visual_blocks_npu_*.omc 2>/dev/null | awk '{print "  "$9"  ("$5")"}'
echo ""
echo "快速验证完成: 6 个 chunk 的 OMC 已用 1 个校准样本 (W8A8, DDK 6.1.1.0-complete) 生成完毕。"
echo "最终 OMC 已汇总到: ${OM_COLLECT_DIR}/visual_blocks_npu_{0..5}.omc"

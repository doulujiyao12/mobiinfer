#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 快速验证: 只用 1 个校准样本 (num_samples=1) 跑 W8A8 量化,
# 但导出 fp16 权重的 OM 文件 (不做 int8 量化, 走 plain MatMul)。
#
# 与量化脚本 (run_real_calib_1_W8A8_v6110_omc.sh) 的区别:
#   - ONNX 导出时用 --fp16, 权重转成 fp16 (而非 fp32)
#   - om 编译不带 --compress_conf (不 int8 量化)
#   - --target=om (产物 .om, 手机上 JIT 编译)
#   - 算子仍是 plain MatMul (fp16 非量化)
#
# 说明: 这里仍走 visual_plugin_quant_matmul_route.py 的 all 流程,
#   目的是复用其 FLinearMatmul plain-MatMul 的 ONNX 导出逻辑。
#   但因为最终不喂 compress_conf, 量化参数不会被 omg 使用,
#   最终产物是 fp16 权重的非量化 OM。
#
# 校准输入目录复用 /temp/fdh/input_calib_fdh/calib_inputs_16
#   - load_calibration_samples 取前 1 个样本固定 shape
#
# 链路:
#   calib_inputs_16 (fp16 npz, 取 1 个)
#     -> visual_plugin_quant_matmul_route.py (--fp16, 导出 fp16 权重 ONNX)
#     -> run_visual_plugin_matmul_om_fp16.sh (/temp/fdh/model_omc_fdh_9030/model_..._chunk_W8A8_{i}_real1_w8a8_fp16/)
#     -> 6 个 visual_plugin_matmul_quantized.om (fp16 权重)
#     -> 汇总重命名为 visual_blocks_npu_{0..5}.om
# ============================================================

# ============================== 可调参数 ==============================
NUM_SAMPLES=1                   # 只用 1 个样本固定 shape
NPU_CHUNKS=6
PLATFORM=kirin9030

# 复用已生成的 16 样本校准 npz (只取前 1 个)
CALIB_NPZ_DIR=/temp/fdh/input_calib_fdh/calib_inputs_16

OMC_OUT_ROOT=/temp/fdh/model_omc_fdh_9030
ROUTE_SUFFIX=real1_w8a8_fp16    # route_dir 后缀 (独立)
OM_COLLECT_DIR=${OMC_OUT_ROOT}/model_visual_plugin_matmul_W8A8_${ROUTE_SUFFIX}   # 汇总目录
OMC_SCRIPT=run_visual_plugin_matmul_om_fp16.sh
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # mobiinfer

set -e

echo "=========================================="
echo "  快速验证 (W8A8 -> fp16 权重 OM, 1 样本, DDK 6.1.1.0-complete)"
echo "  num_samples  : ${NUM_SAMPLES}"
echo "  weight       : fp16 (非 int8 量化)"
echo "  target       : om (手机上 JIT 编译)"
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

# ---------- 5) 6 chunk 量化 + 导出 ONNX + OM ----------
echo "[step 5] 逐 chunk 跑 导出 ONNX + OM (fp16 权重, num_samples=${NUM_SAMPLES})"
echo ""

# ---- 环境 ----
source /opt/conda/etc/profile.d/conda.sh
conda activate CANN
export DDK_DOPT=/temp/fdh/ddk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3
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

    echo "[chunk ${i}] running prepare + calibrate + export-onnx (fp16 权重, num_samples=${NUM_SAMPLES}) ..."
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
        --fp16 \
        --force_regen \
        all; then
        echo "[chunk ${i}] ERROR: all step 失败"
        FAILED_CHUNKS+=("chunk_${i}:all")
        echo ""
        continue
    fi
    echo "[chunk ${i}] ONNX 已导出 (fp16 权重): ${ONNX_FILE}"

    echo "[chunk ${i}] running OMG -> om (fp16, ${PLATFORM}) ..."
    if PLATFORM="${PLATFORM}" bash "${OMC_SCRIPT}" \
            "${ROUTE_DIR}" > "${OMC_LOG}" 2>&1; then
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
echo "校准输入: ${CALIB_NPZ_DIR}/ (每 chunk 取前 1 个, 共 6 个样本)"

if [ ${#FAILED_CHUNKS[@]} -gt 0 ]; then
    echo ""
    echo "Failures: ${FAILED_CHUNKS[*]}"
    exit 1
fi
if [ ${ALL_OK} -ne 1 ]; then
    echo "部分 OM 缺失。"
    exit 1
fi

# ---------- 7) 汇总 OM 到单一目录 ----------
echo ""
echo "[step 7] 汇总 OM -> ${OM_COLLECT_DIR}"
mkdir -p "${OM_COLLECT_DIR}"
for i in 0 1 2 3 4 5; do
    SRC_OM="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W8A8_${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.om"
    DST_OM="${OM_COLLECT_DIR}/visual_blocks_npu_${i}.om"
    if [ -f "${SRC_OM}" ]; then
        cp -f "${SRC_OM}" "${DST_OM}"
        echo "  [OK]   visual_blocks_npu_${i}.om  ($(ls -lh "${DST_OM}" 2>/dev/null | awk '{print $5}'))"
    else
        echo "  [FAIL] 缺失源 OM: ${SRC_OM}"
        ALL_OK=0
    fi
done
if [ ${ALL_OK} -ne 1 ]; then
    echo "汇总 OM 缺失, 请检查。"
    exit 1
fi
echo ""
echo "汇总目录: ${OM_COLLECT_DIR}/"
ls -lh "${OM_COLLECT_DIR}"/visual_blocks_npu_*.om 2>/dev/null | awk '{print "  "$9"  ("$5")"}'
echo ""
echo "快速验证完成: 6 个 chunk 的 OM 已用 fp16 权重 (W8A8 流程, DDK 6.1.1.0-complete) 生成完毕。"
echo "最终 OM 已汇总到: ${OM_COLLECT_DIR}/visual_blocks_npu_{0..5}.om"

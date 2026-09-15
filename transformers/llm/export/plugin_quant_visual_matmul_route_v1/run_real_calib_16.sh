#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 一次性流程: 用 16 张真实图像 (从 256 张中均匀抽样) 的校准输入
# 生成全部 6 个 visual chunk 的量化 OM 文件。
#
# 与 run_real_calib_256.sh 的区别:
#   1) DDK 工具链升级到 6.1.1.0
#   2) 校准样本从 256 -> 16 (calib_inputs_256 中每 16 取 1, 共 16 张/样本)
#   3) 复用已生成的 npz (跳过 dump / bin->npz 步骤), 直接做量化+导出OMC
#
# 校准输入目录: /temp/fdh/input_calib_fdh/calib_inputs_16
#   - 每个 chunk 16 个 npz (sample_000, 016, 032, ..., 240)
#   - 由 calib_inputs_256 均匀抽样而来, 6 chunk * 16 = 96 npz
#
# 链路 (仅量化部分):
#   calib_inputs_16 (fp16 npz)
#     -> visual_plugin_quant_matmul_route.py (每 chunk, --num_samples 16)
#     -> run_visual_plugin_matmul_omc.sh    (/temp/fdh/model_omc_fdh/model_..._chunk{i}_real16/)
#     -> 6 个 visual_plugin_matmul_quantized.om
# ============================================================

# ============================== 可调参数 ==============================
NUM_SAMPLES=16                  # 量化用几个校准样本 (16 张)
NPU_CHUNKS=6
PLATFORM=kirin9030

# 复用已生成的 16 样本校准 npz
CALIB_NPZ_DIR=/temp/fdh/input_calib_fdh/calib_inputs_16

OMC_OUT_ROOT=/temp/fdh/model_omc_fdh_9020
ROUTE_SUFFIX=real16             # route_dir 后缀
OM_COLLECT_DIR=${OMC_OUT_ROOT}/model_visual_plugin_matmul_${ROUTE_SUFFIX}   # 汇总目录
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # mobiinfer

set -e

echo "=========================================="
echo "  真实图像校准量化 (16 样本, DDK 6.1.1.0)"
echo "  num_samples  : ${NUM_SAMPLES}"
echo "  calib_npz    : ${CALIB_NPZ_DIR}"
echo "  route_suffix : ${ROUTE_SUFFIX}"
echo "=========================================="
echo ""

mkdir -p "${OMC_OUT_ROOT}"

# ---------- 0) 前置检查 ----------
if [ ! -d "${CALIB_NPZ_DIR}" ]; then
    echo "ERROR: 校准 npz 目录不存在: ${CALIB_NPZ_DIR}"
    echo "       请先从 calib_inputs_256 抽样生成 16 样本目录"
    exit 1
fi
CALIB_NPZ_COUNT=$(ls "${CALIB_NPZ_DIR}"/*.npz 2>/dev/null | wc -l)
echo "[check] 校准 npz 文件数: ${CALIB_NPZ_COUNT} (期望 6*16=96)"
if [ "${CALIB_NPZ_COUNT}" -lt 96 ]; then
    echo "WARN: 校准 npz 数量不足 96, 请检查 ${CALIB_NPZ_DIR}"
fi
echo ""

# ---------- 5) 6 chunk 量化 + 导出 ONNX + OMC ----------
echo "[step 5] 逐 chunk 跑 量化+导出ONNX+OMC (num_samples=${NUM_SAMPLES})"
echo ""

# ---- 环境 ----
source /opt/conda/etc/profile.d/conda.sh
conda activate CANN
export DDK_DOPT=/temp/fdh/ddk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=${DDK_DOPT}:${REPO_ROOT}/transformers/llm/export:${PYTHONPATH:-}

cd "${SCRIPT_DIR}"

FAILED_CHUNKS=()
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk${i}_${ROUTE_SUFFIX}"
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

    echo "[chunk ${i}] running prepare + calibrate + export-onnx (num_samples=${NUM_SAMPLES}) ..."
    if ! python visual_plugin_quant_matmul_route.py \
        --route_dir "${ROUTE_DIR}" \
        --chunk_index ${i} \
        --npu_chunks ${NPU_CHUNKS} \
        --quant_strategy Quant_aigc_ptq \
        --weight_bit 8 \
        --weight_algo min_max \
        --act_bit 16 \
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
    OM_FILE="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.om"
    if [ -f "${OM_FILE}" ]; then
        SIZE=$(ls -lh "${OM_FILE}" 2>/dev/null | awk '{print $5}')
        echo "  [OK]   chunk ${i}: ${OM_FILE}  (${SIZE})"
    else
        echo "  [FAIL] chunk ${i}: 缺失 ${OM_FILE}"
        ALL_OK=0
    fi
done

echo ""
echo "校准输入: ${CALIB_NPZ_DIR}/ ($(ls "${CALIB_NPZ_DIR}"/*.npz 2>/dev/null | wc -l) npz)"

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
    SRC_OM="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.om"
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
echo "全部 6 个 chunk 的 OM 已用真实校准输入 (16 张图, seq_len=608, DDK 6.1.1.0) 生成完毕。"
echo "最终 OM 已汇总到: ${OM_COLLECT_DIR}/visual_blocks_npu_{0..5}.om"

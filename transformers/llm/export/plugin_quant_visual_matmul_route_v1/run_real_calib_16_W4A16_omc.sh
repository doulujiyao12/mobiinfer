#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 一次性流程: 用 16 张真实图像 (从 256 张中均匀抽样) 的校准输入
# 生成全部 6 个 visual chunk 的 W4A16 量化 OMC 文件。
#
# W4A16 = 权重 4bit (grouplinear, group128) + 激活 16bit
#   - quant_strategy = Quant_act_weight_eco  (官方文档表1: 16-4grouplinear 策略)
#   - weight_bit=4, group_size=128
#   - act_bit=16 (signed)
#
# 与 run_real_calib_16_W4A16.sh 的区别: 本脚本用 --target=omc,
# 产物后缀为 .omc (run_visual_plugin_matmul_omc_omc.sh)。
#
# DDK 量化工具: 6.1.1.0
# OMG 编译  : run_visual_plugin_matmul_omc_omc.sh (--target=omc -> 产物 .omc)
# 平台      : kirin9030
# 校准样本  : 16 (calib_inputs_256 中每 16 取 1, 共 16 张/样本)
# 复用已生成的 npz, 直接做量化+导出OMC
#
# 校准输入目录: /temp/fdh/input_calib_fdh/calib_inputs_16
# 链路 (仅量化部分):
#   calib_inputs_16 (fp16 npz)
#     -> visual_plugin_quant_matmul_route.py (每 chunk, W4A16)
#     -> run_visual_plugin_matmul_omc_omc.sh (/temp/fdh/model_omc_fdh/model_..._chunk_W4A16_omc_{i}_real16_w4a16_omc/)
#     -> 6 个 visual_plugin_matmul_quantized.omc
#     -> 汇总重命名为 visual_blocks_npu_{0..5}.omc
# ============================================================

# ============================== 可调参数 ==============================
NUM_SAMPLES=16                  # 量化用几个校准样本 (16 张)
NPU_CHUNKS=6
PLATFORM=kirin9030

# 复用已生成的 16 样本校准 npz
CALIB_NPZ_DIR=/temp/fdh/input_calib_fdh/calib_inputs_16

OMC_OUT_ROOT=/temp/fdh/model_omc_fdh
ROUTE_SUFFIX=real16_w4a16_omc   # route_dir 后缀
OM_COLLECT_DIR=${OMC_OUT_ROOT}/model_visual_plugin_matmul_W4A16_omc_${ROUTE_SUFFIX}   # 汇总目录
OMC_SCRIPT=run_visual_plugin_matmul_omc_omc.sh
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # mobiinfer

set -e

echo "=========================================="
echo "  真实图像校准量化 (W4A16 -> OMC, 16 样本, DDK 6.1.1.0)"
echo "  quant_strategy: Quant_act_weight_eco"
echo "  weight_bit    : 4 (group_size 128)"
echo "  act_bit       : 16 (signed)"
echo "  target        : omc (产物后缀 .omc)"
echo "  platform      : ${PLATFORM}"
echo "  num_samples   : ${NUM_SAMPLES}"
echo "  calib_npz     : ${CALIB_NPZ_DIR}"
echo "  route_suffix  : ${ROUTE_SUFFIX}"
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
if [ ! -f "${SCRIPT_DIR}/${OMC_SCRIPT}" ]; then
    echo "ERROR: 缺少 OMC 编译脚本: ${SCRIPT_DIR}/${OMC_SCRIPT}"
    exit 1
fi
echo ""

# ---------- 5) 6 chunk 量化 + 导出 ONNX + OMC ----------
echo "[step 5] 逐 chunk 跑 量化+导出ONNX+OMC (W4A16 -> omc, num_samples=${NUM_SAMPLES})"
echo ""

# ---- 环境 ----
source /opt/conda/etc/profile.d/conda.sh
conda activate CANN
export DDK_DOPT=/temp/fdh/ddk/DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=${DDK_DOPT}:${REPO_ROOT}/transformers/llm/export:${PYTHONPATH:-}

cd "${SCRIPT_DIR}"

FAILED_CHUNKS=()
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W4A16_omc_${i}_${ROUTE_SUFFIX}"
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

    echo "[chunk ${i}] running prepare + calibrate + export-onnx (W4A16, num_samples=${NUM_SAMPLES}) ..."
    if ! python visual_plugin_quant_matmul_route.py \
        --route_dir "${ROUTE_DIR}" \
        --chunk_index ${i} \
        --npu_chunks ${NPU_CHUNKS} \
        --quant_strategy Quant_act_weight_eco \
        --weight_bit 4 \
        --group_size 128 \
        --act_bit 16 \
        --num_samples ${NUM_SAMPLES} \
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
    OMC_FILE="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W4A16_omc_${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.omc"
    if [ -f "${OMC_FILE}" ]; then
        SIZE=$(ls -lh "${OMC_FILE}" 2>/dev/null | awk '{print $5}')
        echo "  [OK]   chunk ${i}: ${OMC_FILE}  (${SIZE})"
    else
        echo "  [FAIL] chunk ${i}: 缺失 ${OMC_FILE}"
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
    echo "部分 OMC 缺失。"
    exit 1
fi

# ---------- 7) 汇总 OMC 到单一目录 ----------
echo ""
echo "[step 7] 汇总 OMC -> ${OM_COLLECT_DIR}"
mkdir -p "${OM_COLLECT_DIR}"
for i in 0 1 2 3 4 5; do
    SRC_OMC="${OMC_OUT_ROOT}/model_visual_plugin_matmul_chunk_W4A16_omc_${i}_${ROUTE_SUFFIX}/omc_output/visual_plugin_matmul_quantized.omc"
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
echo "全部 6 个 chunk 的 OMC 已用真实校准输入 (W4A16, 16 张图, seq_len=608, DDK 6.1.1.0) 生成完毕。"
echo "最终 OMC 已汇总到: ${OM_COLLECT_DIR}/visual_blocks_npu_{0..5}.omc"

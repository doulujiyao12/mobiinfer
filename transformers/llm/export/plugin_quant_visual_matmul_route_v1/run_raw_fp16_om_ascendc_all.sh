#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 对比实验: 同样用 **AscendC + raw-fp16 权重**，但改用
#            --target=om（在线 IR，tiling 留在设备侧 JIT）
#         而不是 --target=omc（离线固化 tiling）。
#
# 目的: 验证「9030 慢是因为 omc 在 PC 侧把 tiling 固化成单核
#       (usedCoreNum=1 / blockDim==1)」这个假设。
#
# 复用已有 route 目录里的 fp16 ONNX（与 target 无关），
# 只重跑 OMG，产物落到独立的 om_ascendc/ 前缀，不覆盖 .omc。
#
# 对照物:
#   <route>/omc_output/visual_plugin_matmul_quantized.omc   (omc, 700ms)
#   <新的> .../om_ascendc/visual_plugin_matmul_quantized.om (om, 待测)
# ============================================================

NPU_CHUNKS=6
PLATFORM=kirin9030
DDK_ROOT=${DDK_ROOT:-/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete}
OUT_ROOT=${OUT_ROOT:-/temp/fdh/model_omc_fdh_9030}
ROUTE_PREFIX=${ROUTE_PREFIX:-model_visual_rawfp16_omc_chunk}
COLLECT_DIR=${COLLECT_DIR:-${OUT_ROOT}/model_visual_rawfp16_om_chunks}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  raw-fp16 + AscendC + --target=om  (6 chunks)"
echo "  DDK      : ${DDK_ROOT}"
echo "  platform : ${PLATFORM}"
echo "  collect  : ${COLLECT_DIR}"
echo "=========================================="
echo ""

mkdir -p "${COLLECT_DIR}"

FAILED=()
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="${OUT_ROOT}/${ROUTE_PREFIX}_${i}"
    ONNX_FILE="${ROUTE_DIR}/onnx/visual_blocks_npu_${i}.onnx"
    # 独立输出目录，避免和已有 .omc 混淆
    OUT_PREFIX="${ROUTE_DIR}/om_output_ascendc/visual_plugin_matmul_quantized"
    OM_FILE="${OUT_PREFIX}.om"
    OMG_LOG="${ROUTE_DIR}/om_output_ascendc/om_${PLATFORM}.log"

    echo "=========================================="
    echo "  chunk ${i} / $((NPU_CHUNKS - 1))"
    echo "=========================================="

    if [ ! -s "${ONNX_FILE}" ]; then
        echo "[chunk ${i}] ERROR: 缺少 ONNX（请先跑 run_raw_fp16_omc_all.sh 生成）"
        FAILED+=("chunk_${i}:no_onnx")
        continue
    fi

    if [ -s "${OM_FILE}" ]; then
        echo "[chunk ${i}] OM 已存在, 跳过"
    else
        mkdir -p "$(dirname "${OUT_PREFIX}")"
        echo "[chunk ${i}] OMG --target=om + AscendC (无 compress_conf) ..."
        # LOAD_ASCENDC_ENV=true 强制加载 AscendC（脚本默认只在 omc 时自动加载）
        if ! PLATFORM="${PLATFORM}" \
             TARGET_MODEL_TYPE=om \
             USE_COMPRESS_CONF=false \
             LOAD_ASCENDC_ENV=true \
             DDK_ROOT="${DDK_ROOT}" \
             OMG_TOOL="${DDK_ROOT}/tools/tools_omg/omg" \
             OMG_MASTER_DIR="${DDK_ROOT}/tools/tools_omg/master" \
             OUTPUT_PREFIX="${OUT_PREFIX}" \
             bash run_visual_plugin_matmul_omc.sh "${ROUTE_DIR}" fp16 \
                 > "${OMG_LOG}" 2>&1; then
            echo "[chunk ${i}] ERROR: OMG 失败, 日志: ${OMG_LOG}"
            tail -5 "${OMG_LOG}" 2>/dev/null | sed 's/^/    /'
            FAILED+=("chunk_${i}:omg")
            continue
        fi
        if [ ! -s "${OM_FILE}" ]; then
            echo "[chunk ${i}] ERROR: 无产物 ${OM_FILE}"
            FAILED+=("chunk_${i}:missing")
            continue
        fi
        echo "[chunk ${i}] OM OK: $(ls -lh "${OM_FILE}" | awk '{print $5}')"
    fi

    cp -f "${OM_FILE}" "${COLLECT_DIR}/visual_blocks_npu_${i}.om"
    echo "[chunk ${i}] -> ${COLLECT_DIR}/visual_blocks_npu_${i}.om"
    echo ""
done

echo "=========================================="
echo "  汇总: ${COLLECT_DIR}"
echo "=========================================="
ALL_OK=1
for i in 0 1 2 3 4 5; do
    f="${COLLECT_DIR}/visual_blocks_npu_${i}.om"
    if [ -s "$f" ]; then
        echo "  [OK]   $(ls -lh "$f" | awk '{print $5}')  $f"
    else
        echo "  [FAIL] 缺失 $f"; ALL_OK=0
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "Failures: ${FAILED[*]}"; exit 1
fi
[ ${ALL_OK} -ne 1 ] && exit 1
echo ""
echo "完成。产物: ${COLLECT_DIR}/visual_blocks_npu_{0..5}.om"
echo "下一步: python3 check_rawfp16_omc_ops.py ${COLLECT_DIR}"

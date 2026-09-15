#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 生成 6 个 visual chunk 的 **raw-fp16 + 真离线 OMC** 产物。
#
# 与 run_real_calib_1_W8A8_fp16.sh 的关键区别（那份产物很慢的原因）:
#   1. 权重来自原始 HuggingFace 浮点权重 (export-fp16)，不经过
#      DOPT fake_quant_weight —— 避免伪量化/反量化改变数值。
#   2. 用 --target=omc（真离线编译），不是 --target=om（设备侧 JIT IR）。
#   3. **显式 source DDK 的 set_ascendc_env.sh**（Kirin9030 必须），
#      否则 AscendC 算子内核缺失，MatMul 回退到慢路径。
#
# 链路:
#   /temp/models/mobi0402_2B_halfimage_rl (HF safetensors, 原始 fp32)
#     -> visual_plugin_quant_matmul_route.py --fp16 export-fp16
#        -> <route_dir>/onnx/visual_blocks_npu_<i>.onnx  (fp16 权重)
#     -> run_visual_plugin_matmul_omc.sh  (TARGET_MODEL_TYPE=omc, 无 compress_conf)
#        -> <route_dir>/omc_output/visual_plugin_matmul_quantized.omc
#     -> 汇总为 visual_blocks_npu_<i>.om
# ============================================================

# ------------------------- 可调参数 -------------------------
NPU_CHUNKS=6
PLATFORM=kirin9030
DDK_ROOT=${DDK_ROOT:-/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete}
HF_MODEL=${HF_MODEL:-/temp/models/mobi0402_2B_halfimage_rl}
SEQ_LEN=${SEQ_LEN:-608}
HIDDEN=${HIDDEN:-1024}
ROTARY=${ROTARY:-64}

OUT_ROOT=${OUT_ROOT:-/temp/fdh/model_omc_fdh_9030}
ROUTE_PREFIX=${ROUTE_PREFIX:-model_visual_rawfp16_omc_chunk}
COLLECT_DIR=${COLLECT_DIR:-${OUT_ROOT}/model_visual_rawfp16_omc_6chunks}
# -----------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=========================================="
echo "  raw-fp16 + 真离线 OMC  (6 chunks)"
echo "  HF model    : ${HF_MODEL}"
echo "  DDK         : ${DDK_ROOT}"
echo "  platform    : ${PLATFORM}"
echo "  seq/head/dim: ${SEQ_LEN} / ${ROTARY} / ${HIDDEN}"
echo "  out root    : ${OUT_ROOT}"
echo "=========================================="
echo ""

for v in HF_MODEL DDK_ROOT; do
    if [ ! -e "${!v}" ]; then echo "ERROR: $v 不存在: ${!v}"; exit 1; fi
done

mkdir -p "${OUT_ROOT}" "${COLLECT_DIR}"

# ---------- 环境: conda CANN ----------
source /opt/conda/etc/profile.d/conda.sh
conda activate CANN
export PYTHONPATH="${REPO_ROOT}/transformers/llm/export:${PYTHONPATH:-}"

# ---------- 环境: 显式加载 AscendC（关键！）----------
ASCENDC_ENV="${DDK_ROOT}/tools/tools_ascendc/set_ascendc_env.sh"
if [ ! -f "${ASCENDC_ENV}" ]; then
    echo "ERROR: 找不到 AscendC 环境脚本: ${ASCENDC_ENV}"
    exit 1
fi
echo "[env] source AscendC: ${ASCENDC_ENV}"
set +e; set +u
source "${ASCENDC_ENV}"
ASC_STATUS=$?
set -u; set +e
if [ ${ASC_STATUS} -ne 0 ]; then
    echo "ERROR: AscendC 环境加载失败 (status=${ASC_STATUS})"
    exit 1
fi
export DDK_PATH="${DDK_ROOT}"
export TOOLCHAIN_HOME="${DDK_ROOT}"

if ! python -c 'import te_fusion' >/dev/null 2>&1; then
    echo "WARN: te_fusion 不可用 —— OMC 编译会缺少 AscendC 内核。"
    echo "      需要把 DDK 的 ascendc_adapter wheel 装进 tools_ascendc/package/python:"
    echo "      python -m pip install --no-index --no-deps \\"
    echo "        --target '${DDK_ROOT}/tools/tools_ascendc/package/python' \\"
    echo "        '${DDK_ROOT}/tools/tools_ascendc/package/ascendc_adapter-0.1-py3-none-any.whl'"
else
    echo "[env] te_fusion OK"
fi

cd "${SCRIPT_DIR}"

FAILED=()
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="${OUT_ROOT}/${ROUTE_PREFIX}_${i}"
    ONNX_FILE="${ROUTE_DIR}/onnx/visual_blocks_npu_${i}.onnx"
    OMC_FILE="${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized.omc"
    OMC_LOG="${ROUTE_DIR}/omc_output/omc_${PLATFORM}.log"

    echo "=========================================="
    echo "  chunk ${i} / $((NPU_CHUNKS - 1))   route: ${ROUTE_DIR}"
    echo "=========================================="

    # ---------- 1) 导出 raw-fp16 ONNX ----------
    if [ -f "${ONNX_FILE}" ]; then
        echo "[chunk ${i}] ONNX 已存在, 跳过导出"
    else
        echo "[chunk ${i}] export-fp16 (原始 HF 权重 -> fp16 ONNX) ..."
        if ! python visual_plugin_quant_matmul_route.py \
                --route_dir "${ROUTE_DIR}" \
                --chunk_index ${i} \
                --npu_chunks ${NPU_CHUNKS} \
                --model_path "${HF_MODEL}" \
                --sequence_length ${SEQ_LEN} \
                --hidden_size ${HIDDEN} \
                --rotary_size ${ROTARY} \
                --fp16 \
                export-fp16; then
            echo "[chunk ${i}] ERROR: export-fp16 失败"
            FAILED+=("chunk_${i}:export")
            continue
        fi
        echo "[chunk ${i}] ONNX OK: ${ONNX_FILE}"
    fi

    # ---------- 2) OMC 编译 (--target=omc, 无 compress_conf) ----------
    if [ -s "${OMC_FILE}" ]; then
        echo "[chunk ${i}] OMC 已存在, 跳过编译"
    else
        echo "[chunk ${i}] OMG --target=omc (AscendC, 无 compress_conf) ..."
        if ! PLATFORM="${PLATFORM}" \
             TARGET_MODEL_TYPE=omc \
             USE_COMPRESS_CONF=false \
             LOAD_ASCENDC_ENV=true \
             DDK_ROOT="${DDK_ROOT}" \
             OMG_TOOL="${DDK_ROOT}/tools/tools_omg/omg" \
             OMG_MASTER_DIR="${DDK_ROOT}/tools/tools_omg/master" \
             bash run_visual_plugin_matmul_omc.sh "${ROUTE_DIR}" fp16 \
                 > "${OMC_LOG}" 2>&1; then
            echo "[chunk ${i}] ERROR: OMC 编译失败, 日志: ${OMC_LOG}"
            FAILED+=("chunk_${i}:omc")
            continue
        fi
        if [ ! -s "${OMC_FILE}" ]; then
            echo "[chunk ${i}] ERROR: OMC 完成但没有产物: ${OMC_FILE}"
            FAILED+=("chunk_${i}:missing")
            continue
        fi
        echo "[chunk ${i}] OMC OK: $(ls -lh "${OMC_FILE}" | awk '{print $5}')"
    fi

    # ---------- 3) 汇总（改名成 app 识别的 .om）----------
    cp -f "${OMC_FILE}" "${COLLECT_DIR}/visual_blocks_npu_${i}.om"
    echo "[chunk ${i}] -> ${COLLECT_DIR}/visual_blocks_npu_${i}.om"
    echo ""
done

# ---------- 汇总结果 ----------
echo "=========================================="
echo "  汇总目录: ${COLLECT_DIR}"
echo "=========================================="
ALL_OK=1
for i in 0 1 2 3 4 5; do
    f="${COLLECT_DIR}/visual_blocks_npu_${i}.om"
    if [ -s "$f" ]; then
        echo "  [OK]   $(ls -lh "$f" | awk '{print $5}')  $f"
    else
        echo "  [FAIL] 缺失 $f"
        ALL_OK=0
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failures: ${FAILED[*]}"
    exit 1
fi
[ ${ALL_OK} -ne 1 ] && exit 1

echo ""
echo "完成。下一步: 用 check_rawfp16_omc_ops.py 校验算子标记，"
echo "          再把 ${COLLECT_DIR}/visual_blocks_npu_*.om 放入模型目录的 om/ 下。"

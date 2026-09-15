#!/usr/bin/env bash
set -euo pipefail

# 用 complete DDK 6.1.1.0-complete 的 omg 工具导出 OMC 文件 (--target=omc, 产物 .omc)
# 真正把 W8A8 量化落到 NPU 量化算子 (QuantBatchMatmulV3 等)。
#
# 与原 run_visual_plugin_matmul_omc_omc.sh 的区别:
#   - 用 complete 环境 (DDK-tools-next-6.1.1.0-complete)
#   - 显式设置 omc 编译必需的环境变量 (DDK_PATH / PYTHONPATH / LD_LIBRARY_PATH / PATH)
#
# 用法: PLATFORM=kirin9030 bash run_visual_plugin_matmul_omc_v6110_omc.sh <route_dir> [fp16|fp32]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <route_dir> [fp16|fp32]"
    exit 1
fi

DDK_ROOT=${DDK_ROOT:-/temp/fdh/ddk/DDK-tools-next-6.1.1.0-complete}

ROUTE_DIR=$(realpath "$1")
WEIGHT_DTYPE=${2:-fp16}
OMG_TOOL=${OMG_TOOL:-${DDK_ROOT}/tools/tools_omg/omg}
OMG_MASTER_DIR=${OMG_MASTER_DIR:-${DDK_ROOT}/tools/tools_omg/master}
PLATFORM=${PLATFORM:-kirin9030}
SAVE_WEIGHTS_AS_EXTERNAL_DATA=${SAVE_WEIGHTS_AS_EXTERNAL_DATA:-false}
PLATFORM_PLUGIN_DIR=$(dirname "${OMG_TOOL}")/../platform/${PLATFORM}

# ---- omc 编译必需的环境变量 (complete 环境) ----
export DDK_PATH=${DDK_PATH:-${DDK_ROOT}}
export TOOLCHAIN_HOME=${TOOLCHAIN_HOME:-${DDK_ROOT}}
export PYTHONPATH=${DDK_ROOT}/tools/tools_ascendc/package/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${OMG_MASTER_DIR}/lib64:${PLATFORM_PLUGIN_DIR}/lib64:${LD_LIBRARY_PATH:-}
export PATH=${OMG_MASTER_DIR}:${DDK_ROOT}/tools/tools_ascendc/package:${PATH}

ROUTE_CONFIG=${ROUTE_DIR}/route_config.json
MODEL=$(python - <<'PY' "${ROUTE_CONFIG}"
import json, sys
cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print(cfg['model'])
PY
)
INPUT_SHAPE=$(python - <<'PY' "${ROUTE_CONFIG}"
import json, sys
cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print(cfg['input_shape'])
PY
)
QUANT_PARAMS=${ROUTE_DIR}/quant_output/quant_params_file
OUTPUT_PREFIX=${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized

mkdir -p "${ROUTE_DIR}/omc_output"

omg_args=(
  --model "${MODEL}"
  --framework 5
  --output "${OUTPUT_PREFIX}"
  --input_shape="${INPUT_SHAPE}"
  --weight_data_type="${WEIGHT_DTYPE^^}"
  --save_weights_as_external_data="${SAVE_WEIGHTS_AS_EXTERNAL_DATA}"
  --target=omc
  --compress_conf "${QUANT_PARAMS}"
)

if [ -n "${PLATFORM}" ] && [ -d "${PLATFORM_PLUGIN_DIR}" ]; then
  omg_args+=(--platform="${PLATFORM}")
else
  echo "WARN: platform plugin not found, skip --platform"
  echo "      expected: ${PLATFORM_PLUGIN_DIR}"
fi

"${OMG_TOOL}" "${omg_args[@]}"

echo "Done: ${OUTPUT_PREFIX}.omc"

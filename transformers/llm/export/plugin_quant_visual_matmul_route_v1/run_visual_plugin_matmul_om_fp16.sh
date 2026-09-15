#!/usr/bin/env bash
set -euo pipefail

# 用 complete DDK 6.1.1.0-complete 的 omg 工具导出 fp16 权重的 OM 文件。
# 与量化脚本的区别:
#   - 不带 --compress_conf (不 int8 量化权重)
#   - --weight_data_type=FP16 (权重从 fp32 转成 fp16)
#   - --target=om (产物 .om, 手机上 JIT 编译)
#   - 算子仍是 plain MatMul (走 fp16 非量化路径)
#
# 用法: PLATFORM=kirin9030 bash run_visual_plugin_matmul_om_fp16.sh <route_dir>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <route_dir>"
    exit 1
fi

DDK_ROOT=${DDK_ROOT:-/temp/fdh/ddk/DDK-tools-next-6.1.1.0}

ROUTE_DIR=$(realpath "$1")
OMG_TOOL=${OMG_TOOL:-${DDK_ROOT}/tools/tools_omg/omg}
OMG_MASTER_DIR=${OMG_MASTER_DIR:-${DDK_ROOT}/tools/tools_omg/master}
PLATFORM=${PLATFORM:-kirin9030}
SAVE_WEIGHTS_AS_EXTERNAL_DATA=${SAVE_WEIGHTS_AS_EXTERNAL_DATA:-false}
PLATFORM_PLUGIN_DIR=$(dirname "${OMG_TOOL}")/../platform/${PLATFORM}

# ---- omg 运行环境 (complete 环境) ----
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
OUTPUT_PREFIX=${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized

mkdir -p "${ROUTE_DIR}/omc_output"

# 注意: 不带 --compress_conf, 权重走 fp16 而非 int8 量化
omg_args=(
  --model "${MODEL}"
  --framework 5
  --output "${OUTPUT_PREFIX}"
  --input_shape="${INPUT_SHAPE}"
  --weight_data_type=FP16
  --save_weights_as_external_data="${SAVE_WEIGHTS_AS_EXTERNAL_DATA}"
  --target=om
)

if [ -n "${PLATFORM}" ] && [ -d "${PLATFORM_PLUGIN_DIR}" ]; then
  omg_args+=(--platform="${PLATFORM}")
else
  echo "WARN: platform plugin not found, skip --platform"
  echo "      expected: ${PLATFORM_PLUGIN_DIR}"
fi

"${OMG_TOOL}" "${omg_args[@]}"

echo "Done: ${OUTPUT_PREFIX}.om"

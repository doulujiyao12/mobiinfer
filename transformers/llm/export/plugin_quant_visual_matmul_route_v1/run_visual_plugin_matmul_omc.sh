#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <route_dir> [fp16|fp32]"
    exit 1
fi

ROUTE_DIR=$(realpath "$1")
WEIGHT_DTYPE=${2:-fp16}
OMG_TOOL=${OMG_TOOL:-/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_omg/omg}
OMG_MASTER_DIR=${OMG_MASTER_DIR:-/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_omg/master}
PLATFORM=${PLATFORM:-kirinx90}
# Preserve the original online-IR behavior unless the caller explicitly asks
# for a true offline-compiled OMC artifact.
TARGET_MODEL_TYPE=${TARGET_MODEL_TYPE:-om}
USE_COMPRESS_CONF=${USE_COMPRESS_CONF:-true}
LOAD_ASCENDC_ENV=${LOAD_ASCENDC_ENV:-auto}
SAVE_WEIGHTS_AS_EXTERNAL_DATA=${SAVE_WEIGHTS_AS_EXTERNAL_DATA:-false}
PLATFORM_PLUGIN_DIR=$(dirname "${OMG_TOOL}")/../platform/${PLATFORM}
ASCENDC_ENV_SCRIPT=${ASCENDC_ENV_SCRIPT:-$(dirname "${OMG_TOOL}")/../tools_ascendc/set_ascendc_env.sh}

if [ "${TARGET_MODEL_TYPE}" != "omc" ] && [ "${TARGET_MODEL_TYPE}" != "om" ]; then
  echo "ERROR: TARGET_MODEL_TYPE must be 'omc' (offline compiled) or 'om' (online IR), got: ${TARGET_MODEL_TYPE}"
  exit 1
fi
if [ "${USE_COMPRESS_CONF}" != "true" ] && [ "${USE_COMPRESS_CONF}" != "false" ]; then
  echo "ERROR: USE_COMPRESS_CONF must be 'true' or 'false', got: ${USE_COMPRESS_CONF}"
  exit 1
fi
if [ "${LOAD_ASCENDC_ENV}" != "auto" ] && [ "${LOAD_ASCENDC_ENV}" != "true" ] && \
    [ "${LOAD_ASCENDC_ENV}" != "false" ]; then
  echo "ERROR: LOAD_ASCENDC_ENV must be 'auto', 'true', or 'false', got: ${LOAD_ASCENDC_ENV}"
  exit 1
fi

should_load_ascendc=false
if [ "${LOAD_ASCENDC_ENV}" = "true" ]; then
  should_load_ascendc=true
elif [ "${LOAD_ASCENDC_ENV}" = "auto" ] && [ "${PLATFORM}" = "kirin9030" ] && \
    [ "${TARGET_MODEL_TYPE}" = "omc" ]; then
  should_load_ascendc=true
fi

if [ "${should_load_ascendc}" = "true" ]; then
  if [ ! -f "${ASCENDC_ENV_SCRIPT}" ]; then
    echo "ERROR: Kirin 9030 offline compilation requires the AscendC environment: ${ASCENDC_ENV_SCRIPT}"
    exit 1
  fi
  # Huawei's script uses non-zero grep results for its source/execute check, so
  # temporarily relax the caller's strict flags while importing the environment.
  set +e
  set +u
  source "${ASCENDC_ENV_SCRIPT}"
  ascendc_status=$?
  set -u
  set -e
  if [ ${ascendc_status} -ne 0 ]; then
    echo "ERROR: failed to load the AscendC environment: ${ASCENDC_ENV_SCRIPT}"
    exit ${ascendc_status}
  fi
  if ! python -c 'import te_fusion' >/dev/null 2>&1; then
    echo "ERROR: AscendC environment loaded but te_fusion is unavailable."
    echo "       Install the DDK-provided ascendc_adapter wheel into tools_ascendc/package/python."
    exit 1
  fi
fi

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
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized}
OUTPUT_FILE=${OUTPUT_PREFIX}.${TARGET_MODEL_TYPE}

mkdir -p "${ROUTE_DIR}/omc_output"
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"
export PATH="${OMG_MASTER_DIR}:${PATH}"

omg_args=(
  --model "${MODEL}"
  --framework 5
  --output "${OUTPUT_PREFIX}"
  --input_shape="${INPUT_SHAPE}"
  --weight_data_type="${WEIGHT_DTYPE^^}"
  --save_weights_as_external_data="${SAVE_WEIGHTS_AS_EXTERNAL_DATA}"
  --target="${TARGET_MODEL_TYPE}"
)

if [ "${USE_COMPRESS_CONF}" = "true" ]; then
  if [ ! -s "${QUANT_PARAMS}" ]; then
    echo "ERROR: quantization config is missing or empty: ${QUANT_PARAMS}"
    exit 1
  fi
  omg_args+=(--compress_conf "${QUANT_PARAMS}")
else
  echo "INFO: compiling without --compress_conf (uncompressed ${WEIGHT_DTYPE} offline graph)"
fi

if [ -n "${PLATFORM}" ] && [ -d "${PLATFORM_PLUGIN_DIR}" ]; then
  omg_args+=(--platform="${PLATFORM}")
else
  echo "WARN: platform plugin not found, skip --platform"
  echo "      expected: ${PLATFORM_PLUGIN_DIR}"
fi

"${OMG_TOOL}" "${omg_args[@]}"

if [ ! -s "${OUTPUT_FILE}" ]; then
  echo "ERROR: OMG finished without the expected ${TARGET_MODEL_TYPE} artifact: ${OUTPUT_FILE}"
  exit 1
fi

echo "Done: ${OUTPUT_FILE}"

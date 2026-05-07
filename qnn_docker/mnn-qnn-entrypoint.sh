#!/usr/bin/env bash
set -e

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/opt/qnn}"
export PATH="/root/bin/qnn-shim:${PATH}"

if [[ "${BUILD_QNN_MIRROR:-0}" == "1" ]]; then
  /usr/local/bin/build_qnn_mirror.sh
  export QNN_MIRROR="${QNN_MIRROR:-/root/qnn-mirror}"
fi

if [ -d "${QNN_SDK_ROOT}/lib/x86_64-linux-clang" ]; then
  export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
fi

exec "$@"

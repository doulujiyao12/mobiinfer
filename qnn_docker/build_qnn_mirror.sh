#!/usr/bin/env bash
set -euo pipefail

SOURCE_QNN_ROOT="${1:-${QNN_SDK_ROOT:-/opt/qnn}}"
MIRROR_ROOT="${2:-${QNN_MIRROR:-/root/qnn-mirror}}"

MAKEFILE_RELATIVE_PATH="share/QNN/converter/Makefile.linux-x86_64"
MAKEFILE_PATH="${MIRROR_ROOT}/${MAKEFILE_RELATIVE_PATH}"
TEMP_MAKEFILE_PATH="${MAKEFILE_PATH}.tmp"

if [[ ! -d "${SOURCE_QNN_ROOT}" ]]; then
  echo "[build_qnn_mirror] source QNN SDK not found: ${SOURCE_QNN_ROOT}" >&2
  exit 1
fi

if [[ -e "${MIRROR_ROOT}" ]]; then
  rm -rf "${MIRROR_ROOT}"
fi

mkdir -p "${MIRROR_ROOT}"

echo "[build_qnn_mirror] copying QNN SDK from ${SOURCE_QNN_ROOT} to ${MIRROR_ROOT}"
rsync -aH --delete "${SOURCE_QNN_ROOT}/" "${MIRROR_ROOT}/"

if [[ ! -f "${MAKEFILE_PATH}" ]]; then
  echo "[build_qnn_mirror] Makefile not found: ${MAKEFILE_PATH}" >&2
  exit 1
fi

echo "[build_qnn_mirror] patching ${MAKEFILE_PATH}"
python3 - <<'PY' "${MAKEFILE_PATH}" "${TEMP_MAKEFILE_PATH}"
from pathlib import Path
import sys

source_path = Path(sys.argv[1])
temp_path = Path(sys.argv[2])

old_line = "TARGET_OBJCOPY_CMD := objcopy -I binary -O elf64-x86-64 -B i386:x86-64"
new_line = "TARGET_OBJCOPY_CMD := objcopy -I binary -O elf64-x86-64 -B i386:x86-64 --rename-section .data=.ldata,alloc,load,readonly,data,contents"

content = source_path.read_text()
if old_line not in content:
    raise SystemExit(f"expected line not found in {source_path}")

content = content.replace(old_line, new_line, 1)
temp_path.write_text(content)
PY

mv "${TEMP_MAKEFILE_PATH}" "${MAKEFILE_PATH}"

echo "[build_qnn_mirror] mirror ready: ${MIRROR_ROOT}"
echo "[build_qnn_mirror] patched file: ${MAKEFILE_PATH}"

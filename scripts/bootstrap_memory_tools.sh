#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
TOOL_ROOT="${LUTACC_MEMORY_TOOL_ROOT:-/data3/docker_cache/LUTAcc/tools}"
CACTI_ROOT="${TOOL_ROOT}/cacti"
DRAMPOWER_ROOT="${TOOL_ROOT}/DRAMPower"
mkdir -p "${TOOL_ROOT}"

if [ ! -d "${CACTI_ROOT}/.git" ]; then
  git clone --depth 1 https://github.com/HewlettPackard/cacti.git "${CACTI_ROOT}"
fi
make -C "${CACTI_ROOT}" -j4

if [ ! -d "${DRAMPOWER_ROOT}/.git" ]; then
  git clone --depth 1 --branch v6.0.0 \
    https://github.com/tukl-msd/DRAMPower.git "${DRAMPOWER_ROOT}"
fi
if [ ! -x "${PROJECT_DIR}/.venv/bin/cmake" ]; then
  "${PROJECT_DIR}/.venv/bin/python" -m pip install 'cmake==3.30.5'
fi
"${PROJECT_DIR}/.venv/bin/cmake" \
  -S "${DRAMPOWER_ROOT}" \
  -B "${DRAMPOWER_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release
"${PROJECT_DIR}/.venv/bin/cmake" \
  --build "${DRAMPOWER_ROOT}/build" \
  --parallel 4

test -x "${CACTI_ROOT}/cacti"
test -x "${DRAMPOWER_ROOT}/build/bin/cli"
echo "CACTI and DRAMPower bootstrap PASS"

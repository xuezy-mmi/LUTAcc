#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
BUILD_DIR="${RTL_BUILD_DIR:-${PROJECT_DIR}/build/rtl}"
cd "${PROJECT_DIR}"

if ! command -v verilator >/dev/null 2>&1; then
  echo "Verilator is required for RTL smoke tests." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}/oasis" "${BUILD_DIR}/omni"

verilator -cc --exe --build -Wall -Wno-fatal \
  --top-module oasis_rtl_smoke_top \
  --Mdir "${BUILD_DIR}/oasis" \
  rtl/oasis/oasis_concat.sv \
  rtl/oasis/oasis_index_counter.sv \
  rtl/oasis/oasis_cluster.sv \
  rtl/oasis/orizuru_tree.sv \
  rtl/tb/oasis_rtl_smoke_top.sv \
  "${PROJECT_DIR}/rtl/tb/oasis_smoke.cpp"
"${BUILD_DIR}/oasis/Voasis_rtl_smoke_top"

verilator -cc --exe --build -Wall -Wno-fatal \
  --top-module omni_rtl_smoke_top \
  --Mdir "${BUILD_DIR}/omni" \
  rtl/omni/omni_bqu.sv \
  rtl/omni/omni_lgu.sv \
  rtl/omni/omni_lut_pe.sv \
  rtl/tb/omni_rtl_smoke_top.sv \
  "${PROJECT_DIR}/rtl/tb/omni_smoke.cpp"
"${BUILD_DIR}/omni/Vomni_rtl_smoke_top"

echo "RTL simulation smoke PASS"

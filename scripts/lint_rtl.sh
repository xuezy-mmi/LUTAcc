#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
cd "${PROJECT_DIR}"

if ! command -v verilator >/dev/null 2>&1; then
  echo "Verilator is required for RTL lint." >&2
  exit 1
fi

verilator --lint-only -Wall -Wno-fatal \
  --top-module oasis_rtl_smoke_top \
  rtl/oasis/oasis_concat.sv \
  rtl/oasis/oasis_index_counter.sv \
  rtl/oasis/oasis_cluster.sv \
  rtl/oasis/orizuru_tree.sv \
  rtl/tb/oasis_rtl_smoke_top.sv

verilator --lint-only -Wall -Wno-fatal \
  --top-module oasis_controller \
  rtl/oasis/oasis_controller.sv

verilator --lint-only -Wall -Wno-fatal \
  --top-module oasis_mac_tree \
  rtl/oasis/oasis_mac_tree.sv

verilator --lint-only -Wall -Wno-fatal \
  --unroll-count 4096 \
  --top-module oasis_pe_line \
  rtl/oasis/oasis_concat.sv \
  rtl/oasis/oasis_pe_line.sv

verilator --lint-only -Wall -Wno-fatal \
  --top-module omni_rtl_smoke_top \
  rtl/omni/omni_bqu.sv \
  rtl/omni/omni_lgu.sv \
  rtl/omni/omni_lut_pe.sv \
  rtl/tb/omni_rtl_smoke_top.sv

verilator --lint-only -Wall -Wno-fatal \
  --top-module omni_top \
  rtl/omni/omni_lut_pe.sv \
  rtl/omni/omni_array.sv \
  rtl/omni/omni_controller.sv \
  rtl/omni/omni_top.sv

echo "RTL lint PASS"

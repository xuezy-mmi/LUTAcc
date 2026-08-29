#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
BUILD_DIR="${CHIPYARD_OVERLAY_BUILD_DIR:-${PROJECT_DIR}/build/chipyard-overlay}"
RISCV_GCC="${RISCV_GCC:-/root/chipyard/esp-tools-install/bin/riscv64-unknown-elf-gcc}"
mkdir -p "${BUILD_DIR}"
cd "${PROJECT_DIR}"

verilator -cc --exe --build -Wall -Wno-fatal \
  --top-module LUTAccTimingBlackBox \
  --Mdir "${BUILD_DIR}/verilator" \
  integration/chipyard/src/main/resources/vsrc/LUTAccTimingBlackBox.sv \
  "${PROJECT_DIR}/integration/chipyard/timing_smoke.cpp"
"${BUILD_DIR}/verilator/VLUTAccTimingBlackBox"

if [ ! -x "${RISCV_GCC}" ]; then
  echo "RISC-V cross compiler not found: ${RISCV_GCC}" >&2
  exit 1
fi
"${RISCV_GCC}" \
  -march=rv64imafdc -mabi=lp64d -O2 -ffreestanding \
  -I integration/chipyard/software \
  -c integration/chipyard/software/lutacc_rocc_smoke.c \
  -o "${BUILD_DIR}/lutacc_rocc_smoke.o"
test -s "${BUILD_DIR}/lutacc_rocc_smoke.o"
echo "Chipyard RoCC software compile PASS"

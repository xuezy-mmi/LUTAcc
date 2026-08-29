#!/usr/bin/env bash
set -euo pipefail

TOOL_ROOT="${LUTACC_RTL_TOOL_ROOT:-/data3/docker_cache/LUTAcc/tools}"
YOSYS_ROOT="${TOOL_ROOT}/yosys-0.7"
IVERILOG_ROOT="${TOOL_ROOT}/iverilog-10.1"
DOWNLOAD_DIR="${TOOL_ROOT}/debs"
mkdir -p "${YOSYS_ROOT}" "${IVERILOG_ROOT}" "${DOWNLOAD_DIR}"

download_deb() {
  local package_spec="$1"
  local pattern="$2"
  if ! find "${DOWNLOAD_DIR}" -maxdepth 1 -name "${pattern}" -print -quit | grep -q .; then
    (cd "${DOWNLOAD_DIR}" && apt-get download "${package_spec}")
  fi
}

if [ ! -x "${YOSYS_ROOT}/usr/bin/yosys" ]; then
  download_deb 'yosys=0.7-2' 'yosys_0.7-2_amd64.deb'
  download_deb 'libtcl8.6=8.6.8+dfsg-3' 'libtcl8.6_8.6.8+dfsg-3_amd64.deb'
  download_deb 'berkeley-abc=1.01+20161002hgeb6eca6+dfsg-1' 'berkeley-abc_1.01+20161002hgeb6eca6+dfsg-1_amd64.deb'
  for archive in \
    "${DOWNLOAD_DIR}/yosys_0.7-2_amd64.deb" \
    "${DOWNLOAD_DIR}/libtcl8.6_8.6.8+dfsg-3_amd64.deb" \
    "${DOWNLOAD_DIR}/berkeley-abc_1.01+20161002hgeb6eca6+dfsg-1_amd64.deb"
  do
    dpkg-deb -x "${archive}" "${YOSYS_ROOT}"
  done
fi

if [ ! -x "${IVERILOG_ROOT}/usr/bin/iverilog" ]; then
  download_deb 'iverilog=10.1-0.1build1' 'iverilog_10.1-0.1build1_amd64.deb'
  dpkg-deb -x \
    "${DOWNLOAD_DIR}/iverilog_10.1-0.1build1_amd64.deb" \
    "${IVERILOG_ROOT}"
fi

export LD_LIBRARY_PATH="${YOSYS_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export YOSYS_DATDIR="${YOSYS_ROOT}/usr/share/yosys"
"${YOSYS_ROOT}/usr/bin/yosys" -V
"${IVERILOG_ROOT}/usr/bin/iverilog" \
  -B "${IVERILOG_ROOT}/usr/lib/x86_64-linux-gnu/ivl" \
  -V 2>&1 | sed -n '1,3p'

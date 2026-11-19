#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    echo "ensurepip is unavailable; bootstrapping pip inside an isolated --without-pip venv." >&2
    "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
    GET_PIP_PATH="$(mktemp /tmp/lutacc-get-pip.XXXXXX.py)"
    curl -L --fail --retry 2 \
      -o "${GET_PIP_PATH}" \
      https://bootstrap.pypa.io/pip/3.8/get-pip.py
    "${VENV_DIR}/bin/python" "${GET_PIP_PATH}" 'pip<25' setuptools wheel
    rm -f -- "${GET_PIP_PATH}"
  fi
fi

PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_DIR}/.pip-cache}"
export PIP_CACHE_DIR
"${VENV_DIR}/bin/python" -m pip install --upgrade 'pip<25' setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}[dev]"

"${VENV_DIR}/bin/python" - <<'PY'
import lutacc
import numpy
import yaml

print("lutacc", lutacc.__version__)
print("numpy", numpy.__version__)
print("yaml", yaml.__version__)
PY

if command -v verilator >/dev/null 2>&1; then
  verilator --version
else
  echo "Warning: Verilator is not installed; Python tests remain available." >&2
fi

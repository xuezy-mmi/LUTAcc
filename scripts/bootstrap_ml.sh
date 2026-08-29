#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
VENV_DIR="${ML_VENV_DIR:-${PROJECT_DIR}/.venv-ml}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_DIR}/.pip-cache}"
export PIP_CACHE_DIR

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
    GET_PIP_PATH="$(mktemp /tmp/lutacc-ml-get-pip.XXXXXX.py)"
    curl -L --fail --retry 2 \
      -o "${GET_PIP_PATH}" \
      https://bootstrap.pypa.io/pip/3.8/get-pip.py
    "${VENV_DIR}/bin/python" "${GET_PIP_PATH}" 'pip<25' setuptools wheel
    rm -f -- "${GET_PIP_PATH}"
  fi
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade 'pip<25' setuptools wheel
"${VENV_DIR}/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.4.1' \
  'transformers==4.44.2' \
  'accelerate==0.34.2' \
  'datasets==2.21.0' \
  'lm_eval[hf]==0.4.5' \
  'sentencepiece==0.2.0' \
  'protobuf>=4.25,<6' \
  'safetensors>=0.4.5' \
  'numpy>=1.24,<2.0' \
  'PyYAML>=6.0' \
  'pytest>=8,<9'
"${VENV_DIR}/bin/python" -m pip install --no-deps -e "${PROJECT_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import datasets
import lutacc

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("lutacc", lutacc.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch cannot see a GPU")
PY

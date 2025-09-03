#!/usr/bin/env bash
set -euo pipefail

# Simple GPU install helper for CUDA builds of PyTorch.
# Usage:
#   bash scripts/install_gpu.sh               # defaults to cu121
#   bash scripts/install_gpu.sh 118           # use cu118
#   CUDA=121 bash scripts/install_gpu.sh      # alt env-driven

CUDA_VER=${1:-${CUDA:-121}}
if [[ "$CUDA_VER" != "121" && "$CUDA_VER" != "118" ]]; then
  echo "Unsupported CUDA version '$CUDA_VER'. Use 121 or 118." >&2
  exit 1
fi

CUDA_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VER}"

echo "[install_gpu] Using CUDA cu${CUDA_VER} wheels from: ${CUDA_INDEX_URL}"

# Create and activate venv
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip

# Install CUDA-enabled torch first to avoid CPU wheel being pulled by -r
python -m pip install --index-url "${CUDA_INDEX_URL}" torch -U

# Install project requirements; keep CUDA index as extra for torch-related deps
python -m pip install -r requirements.txt --extra-index-url "${CUDA_INDEX_URL}"

echo "[install_gpu] Installed dependencies with CUDA cu${CUDA_VER}."
echo "[install_gpu] Tip: If using gated models (e.g., Gemma), run 'huggingface-cli login' or set HF_TOKEN."


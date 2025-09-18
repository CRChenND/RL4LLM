#!/usr/bin/env bash
set -Eeuo pipefail

# First-time setup script
# - Creates a virtual environment (default: ./venv)
# - Installs requirements from requirements.txt
# - If sourced, activates the venv in the current shell

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

# Allow overriding via env vars
venv_dir="${VENV_DIR:-venv}"
PY_BIN="${PYTHON_BIN:-}"

echo "[setup] Working directory: $REPO_ROOT"
echo "[setup] Virtualenv dir: $venv_dir"

# Pick a Python without enforcing a specific version
if [[ -z "${PY_BIN}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
  else
    echo "[error] Could not find a Python interpreter on PATH (tried python3.11, python3, python)." >&2
    exit 1
  fi
fi

echo "[setup] Using Python: $PY_BIN ($($PY_BIN -V))"

if [[ ! -d "$venv_dir" ]]; then
  echo "[setup] Creating virtual environment at $venv_dir"
  "$PY_BIN" -m venv "$venv_dir"
else
  echo "[setup] Reusing existing virtual environment at $venv_dir"
fi

"$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ -f requirements.txt ]]; then
  echo "[setup] Installing dependencies from requirements.txt"
  "$venv_dir/bin/pip" install -r requirements.txt
else
  echo "[warn] requirements.txt not found; skipping dependency installation"
fi

# Detect whether the script is being sourced
is_sourced=0
if [[ -n "${ZSH_EVAL_CONTEXT:-}" ]]; then
  [[ $ZSH_EVAL_CONTEXT == *:file ]] && is_sourced=1
elif [[ -n "${BASH_SOURCE:-}" ]]; then
  [[ "${BASH_SOURCE[0]}" != "$0" ]] && is_sourced=1
fi

if [[ $is_sourced -eq 1 ]]; then
  echo "[setup] Activating venv in current shell"
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
else
  echo "[setup] Done. To activate the venv, run:"
  echo "        source $venv_dir/bin/activate"
fi

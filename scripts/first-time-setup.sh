#!/usr/bin/env bash
set -Eeuo pipefail

# First-time setup script
# - Ensures Python 3.11.13 is available
# - Creates a virtual environment at .venv
# - Installs requirements from requirements.txt
# - If sourced, activates the venv in the current shell

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

want_py="3.11.13"
venv_dir=".venv"

echo "[setup] Working directory: $REPO_ROOT"
echo "[setup] Target Python: $want_py"

find_python_311() {
  local py
  if command -v python3.11 >/dev/null 2>&1; then
    py="$(command -v python3.11)"
  elif command -v pyenv >/dev/null 2>&1; then
    if ! pyenv versions --bare | grep -qx "$want_py"; then
      echo "[setup] Python $want_py not found in pyenv. Installing (this may take a while)..."
      pyenv install -s "$want_py"
    fi
    # Resolve the python binary for the requested version via pyenv
    py="$(PYENV_VERSION="$want_py" pyenv which python)"
  elif command -v python3 >/dev/null 2>&1; then
    local ver
    ver="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    if [[ "$ver" == "$want_py" ]]; then
      py="$(command -v python3)"
    else
      echo "[error] Found python3 $ver, but $want_py is required." >&2
      echo "        Install Python $want_py (e.g., via pyenv or your package manager)." >&2
      return 1
    fi
  else
    echo "[error] No suitable Python found. Please install python $want_py or pyenv." >&2
    return 1
  fi
  printf '%s\n' "$py"
}

PY_BIN="$(find_python_311)"
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


#!/usr/bin/env bash
set -Eeuo pipefail

# Activate virtual environment (prefer ./venv, fallback to ./.venv)
if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[error] No virtual environment found at ./venv or ./.venv" >&2
  echo "        Create one first (e.g., run scripts/first-time-setup.sh)" >&2
  exit 1
fi

# Run reinforce training
if [[ "$#" -gt 0 ]]; then
  python -m src.algos.reinforce_train "$@"
else
  python -m src.algos.reinforce_train --config configs/reinforce.yaml
fi


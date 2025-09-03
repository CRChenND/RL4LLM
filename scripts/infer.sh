#!/usr/bin/env bash
set -euo pipefail

# Wrapper for infer/infer.py
# Usage:
#   bash scripts/infer.sh [ADAPTER_DIR] [extra args]
# If ADAPTER_DIR is omitted, picks the latest outputs/*/final dir.

ADAPTER_DIR=${1:-}
NO_ADAPTER=0

if [[ -n "${ADAPTER_DIR}" ]]; then
  case "${ADAPTER_DIR}" in
    --no-adapter|--base|base|base-only)
      NO_ADAPTER=1
      ADAPTER_DIR=""
      shift || true
      ;;
    *)
      shift || true
      ;;
  esac
else
  # If no arg provided, try to auto-pick latest adapter
  ADAPTER_DIR=$(ls -td outputs/*/final 2>/dev/null | head -n 1 || true)
  if [[ -z "${ADAPTER_DIR}" ]]; then
    echo "No adapter dir provided and none found under outputs/*/final." >&2
    echo "Tip: use '--no-adapter' to run the base model only." >&2
    echo "Usage: bash scripts/infer.sh [outputs/your-run/final|--no-adapter] [--model_id ...] [--prompt ...]" >&2
    exit 1
  fi
fi

if [[ ${NO_ADAPTER} -eq 0 && ! -d "${ADAPTER_DIR}" ]]; then
  echo "Adapter directory not found: ${ADAPTER_DIR}" >&2
  exit 1
fi

# Activate venv if present
if [[ -d venv ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

if [[ ${NO_ADAPTER} -eq 1 ]]; then
  python3 -m infer.infer "$@"
else
  python3 -m infer.infer --adapter "${ADAPTER_DIR}" "$@"
fi

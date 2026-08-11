#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 BACKEND ARM CALIBRATION_PT [extra trajmc.compression args]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$1"
ARM="$2"
CALIB="$3"
shift 3

RATIO="${RATIO:-0.8}"
SAVE_PATH="${SAVE_PATH:-${ROOT_DIR}/results/weights/${BACKEND}/${ARM}}"

ARGS=(
  --backend "${BACKEND}"
  --calib "${CALIB}"
  --ratio "${RATIO}"
  --layer_type "${LAYER_TYPE:-all}"
  --decomp "${DECOMP:-cholesky}"
  --save_path "${SAVE_PATH}"
  --batch_size "${BATCH_SIZE:-1}"
  --linalg_device "${LINALG_DEVICE:-cpu}"
)
if [[ -n "${MODEL_PATH:-}" ]]; then
  ARGS+=(--model_path "${MODEL_PATH}")
fi
if [[ -n "${XTX_BUDGET_GB:-}" ]]; then
  ARGS+=(--xtx_budget_gb "${XTX_BUDGET_GB}")
fi

cd "${ROOT_DIR}"
python -m trajmc.compression "${ARGS[@]}" "$@"

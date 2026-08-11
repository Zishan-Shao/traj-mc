#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-llada}"
shift "$(( $# > 0 ? 1 : 0 ))"

CORPUS="${CORPUS:-c4}"
SEED="${SEED:-42}"
SEQLEN="${SEQLEN:-2048}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/calib/${BACKEND}}"

COMMON=(
  --backend "${BACKEND}"
  --corpus "${CORPUS}"
  --seed "${SEED}"
  --seqlen "${SEQLEN}"
  --out_dir "${OUT_DIR}"
)
if [[ -n "${NSAMPLES:-}" ]]; then
  COMMON+=(--nsamples "${NSAMPLES}")
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  COMMON+=(--model_path "${MODEL_PATH}")
fi
if [[ "${CORPUS}" == "c4" && "${C4_STREAMING:-1}" == "1" ]]; then
  COMMON+=(--c4_streaming)
fi

cd "${ROOT_DIR}"
python -m trajmc.calibration "${COMMON[@]}" --arm base "$@"
python -m trajmc.calibration "${COMMON[@]}" --arm ours "$@"

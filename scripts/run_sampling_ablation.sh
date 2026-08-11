#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-llada}"
STAGE="${STAGE:-calibrate}"
CORPUS="${CORPUS:-c4}"
SEED="${SEED:-42}"
SEQLEN="${SEQLEN:-2048}"
PREFIX_RATIO="${PREFIX_RATIO:-0.25}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
CALIB_DIR="${ROOT_DIR}/results/ablation/calib/${BACKEND}"
REPORT_DIR="${ROOT_DIR}/results/ablation/reports/${BACKEND}"
SCHEMES=(random_t grid_t grid_t_prefix rollout)

if [[ "${STAGE}" != "calibrate" && "${STAGE}" != "all" ]]; then
  echo "STAGE must be calibrate or all; got ${STAGE}" >&2
  exit 2
fi

COMMON=(
  --backend "${BACKEND}"
  --corpus "${CORPUS}"
  --seed "${SEED}"
  --seqlen "${SEQLEN}"
  --prefix_ratio "${PREFIX_RATIO}"
  --rollout_steps "${ROLLOUT_STEPS}"
  --out_dir "${CALIB_DIR}"
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
for scheme in "${SCHEMES[@]}"; do
  python -m trajmc.calibration "${COMMON[@]}" --scheme "${scheme}"
done

GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
AUDIT_ARGS=()
declare -A CALIBRATIONS
for scheme in "${SCHEMES[@]}"; do
  path="$(find "${CALIB_DIR}" -maxdepth 1 -type f \
    -name "${BACKEND}_${scheme}_${CORPUS}_n*_s${SEED}_full_${GIT_HASH}_calib.pt" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${path}" ]]; then
    echo "Could not find ${scheme} calibration under ${CALIB_DIR}" >&2
    exit 1
  fi
  CALIBRATIONS["${scheme}"]="${path}"
  AUDIT_ARGS+=(--calib "${scheme}=${path}")
done

mkdir -p "${REPORT_DIR}"
python -m analysis.calibration_ablation \
  "${AUDIT_ARGS[@]}" \
  --out "${REPORT_DIR}/calibration_audit.json"

if [[ "${STAGE}" == "all" ]]; then
  for scheme in "${SCHEMES[@]}"; do
    SAVE_PATH="${ROOT_DIR}/results/ablation/weights/${BACKEND}/${scheme}" \
      "${ROOT_DIR}/scripts/compress_arm.sh" \
      "${BACKEND}" "${scheme}" "${CALIBRATIONS[${scheme}]}"
  done
fi

echo "Ablation calibration audit: ${REPORT_DIR}/calibration_audit.json"

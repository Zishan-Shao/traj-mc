#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "${ROOT_DIR}/scripts/lib.sh"
BACKEND="${1:-llada}"
STAGE="${STAGE:-calibrate}"
CORPUS="${CORPUS:-c4}"
SEED="${SEED:-42}"
SAMPLING_SEED="${SAMPLING_SEED:-${SEED}}"
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
  --sampling_seed "${SAMPLING_SEED}"
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

# Preflight: resolve every target path before any GPU work.  The rollout arm is
# the expensive one, so a naming or configuration problem has to surface here,
# not after it has run.
declare -A CALIBRATIONS
for scheme in "${SCHEMES[@]}"; do
  CALIBRATIONS["${scheme}"]="$(calib_artifact_path "${COMMON[@]}" --scheme "${scheme}")"
  echo "[ablation] ${scheme} -> ${CALIBRATIONS[${scheme}]}"
done

for scheme in "${SCHEMES[@]}"; do
  "${PYTHON_BIN}" -m trajmc.calibration "${COMMON[@]}" --scheme "${scheme}"
  require_calib_artifact "${CALIBRATIONS[${scheme}]}" "${scheme}"
done

AUDIT_ARGS=()
for scheme in "${SCHEMES[@]}"; do
  AUDIT_ARGS+=(--calib "${scheme}=${CALIBRATIONS[${scheme}]}")
done

mkdir -p "${REPORT_DIR}"
"${PYTHON_BIN}" -m analysis.calibration_ablation \
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

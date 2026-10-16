#!/usr/bin/env bash
# Dream-7B-Instruct: calibration and compression at 20% parameter reduction.
#
# Builds the three compressed Dream rows of Table 1 (RATIO = 0.8 retained):
#   nokv_weight_svd  plain SVD of W                         ("SVD")
#   nokv_clean       whitening on clean windows, t = 0      ("SVD-LLM")
#   nokv_traj        whitening on Traj-MC windows, t~U(0,1) ("+Traj-MC")
#
# Layer scope is LAYER_TYPE=nokv: q, o, gate, up, down are factorised and the
# GQA k_proj / v_proj stay dense (140 linears, 80.31% of parameters kept).
# Dream's post-norm MLP covariance is numerically singular in float32, so the
# whitening uses eigh with a relative eigenvalue floor (rcond = 1e-10) instead
# of a plain Cholesky factor.
#
# The eigh of down_proj's 18944 x 18944 covariance needs ~32 GB of GPU memory
# with the model resident; on a smaller card set LINALG_DEVICE=cpu (slower).
#
#   bash scripts/dream_compress.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE="${STAGE:-all}"
RATIO="${RATIO:-0.8}"
LAYER_TYPE="${LAYER_TYPE:-nokv}"
LINALG_DEVICE="${LINALG_DEVICE:-cuda}"
OUT="${RESULTS}/dream_instruct"
CALIB_DIR="${OUT}/calib"
REPORT_DIR="${OUT}/reports"
W="${OUT}/r${RATIO}/weights"
mkdir -p "${CALIB_DIR}" "${REPORT_DIR}"

BACKEND=(--backend dream_instruct --model_path "${DREAM_PATH}")

# ── 1. calibration: the same C4 protocol as LLaDA ────────────────────────────
declare -A CALIB
for scheme in clean_t0 random_t; do
  ARGV=("${BACKEND[@]}" "${CALIB_COMMON[@]}" --scheme "${scheme}"
        --nsamples "${NSAMPLES}" --seed "${SEED}" --sampling_seed "${SEED}"
        --out_dir "${CALIB_DIR}")
  if [[ "${STAGE}" == calibrate || "${STAGE}" == all ]]; then
    CALIB[${scheme}]="$(build_calib "${ARGV[@]}")"
  else
    CALIB[${scheme}]="$(calib_path "${ARGV[@]}")"
  fi
  echo "[dream] ${scheme}: ${CALIB[${scheme}]}"
done
"${PYTHON_BIN}" -m trajmc.analysis.calibration_ablation \
  --calib "clean_t0=${CALIB[clean_t0]}" --calib "random_t=${CALIB[random_t]}" \
  --out "${REPORT_DIR}/calibration_audit.json"

# ── 2. compression ───────────────────────────────────────────────────────────
if [[ "${STAGE}" == compress || "${STAGE}" == all ]]; then
  ARGS=("${BACKEND[@]}" --ratio "${RATIO}" --layer_type "${LAYER_TYPE}"
        --linalg_device "${LINALG_DEVICE}" --batch_size 1 --save_dtype bfloat16)
  compress "${ARGS[@]}" --decomp identity \
    --save_path "${W}/${LAYER_TYPE}_weight_svd" --run_id "dream_r${RATIO}_weight_svd"
  compress "${ARGS[@]}" --decomp eigh --rcond 1e-10 --calib "${CALIB[clean_t0]}" \
    --save_path "${W}/${LAYER_TYPE}_clean" --run_id "dream_r${RATIO}_clean"
  compress "${ARGS[@]}" --decomp eigh --rcond 1e-10 --calib "${CALIB[random_t]}" \
    --save_path "${W}/${LAYER_TYPE}_traj" --run_id "dream_r${RATIO}_traj"
fi
echo "[dream] done -> ${W}"

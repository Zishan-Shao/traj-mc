#!/usr/bin/env bash
# LLaDA-8B-Instruct: calibration, compression, and generation-state error.
#
# Builds the three compressed rows of Table 1 at 20% and 40% parameter
# reduction (RATIO = retained fraction, 0.8 and 0.6):
#   weight_svd  plain SVD of W, no activations                  ("SVD")
#   clean       SVD-LLM whitening on clean windows, t = 0       ("SVD-LLM")
#   traj        SVD-LLM whitening on Traj-MC windows, t~U(0,1)  ("+Traj-MC")
# and E_gen on held-out GSM8K generation states (Fig. 1b).
#
#   bash scripts/llada_compress.sh                 # everything
#   STAGE=calibrate bash scripts/llada_compress.sh # or compress / measure
#
# Outputs under results/llada_instruct/.  Each step skips work already on disk.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE="${STAGE:-all}"
RATIOS="${RATIOS:-0.8 0.6}"
# Whitening solves run on CPU by default (no GPU memory pressure); set
# LINALG_DEVICE=cuda to run them on the card.
LINALG_DEVICE="${LINALG_DEVICE:-cpu}"
EVAL_SEED="${EVAL_SEED:-1337}"   # held-out prompts; must differ from SEED
OUT="${RESULTS}/llada_instruct"
CALIB_DIR="${OUT}/calib"
STATE_DIR="${OUT}/heldout_states"
REPORT_DIR="${OUT}/reports"
mkdir -p "${CALIB_DIR}" "${STATE_DIR}" "${REPORT_DIR}"

BACKEND=(--backend llada_instruct --model_path "${LLADA_PATH}")
calib_argv() {  # scheme
  CALIB_ARGV=("${BACKEND[@]}" "${CALIB_COMMON[@]}" --scheme "$1"
              --nsamples "${NSAMPLES}" --seed "${SEED}" --sampling_seed "${SEED}"
              --out_dir "${CALIB_DIR}")
}
# Held-out evaluation states: real reverse-sampler states of the dense model on
# 256 GSM8K *test* prompts under the official sampler (4-shot, gen 256,
# block 8, 256 steps).  No prompt overlaps any calibration window.
STATE_ARGV=("${BACKEND[@]}" "${CALIB_COMMON[@]}" --scheme rollout
            --rollout_prompts gsm8k_test --nsamples "${NSAMPLES}"
            --seed "${EVAL_SEED}" --sampling_seed "${EVAL_SEED}"
            --rollout_steps 256 --rollout_gen_length 256 --rollout_block_length 8
            --out_dir "${STATE_DIR}")

# ── 1. calibration ───────────────────────────────────────────────────────────
declare -A CALIB
for scheme in clean_t0 random_t; do
  calib_argv "${scheme}"
  if [[ "${STAGE}" == calibrate || "${STAGE}" == all ]]; then
    CALIB[${scheme}]="$(build_calib "${CALIB_ARGV[@]}")"
  else
    CALIB[${scheme}]="$(calib_path "${CALIB_ARGV[@]}")"
  fi
  echo "[llada] ${scheme}: ${CALIB[${scheme}]}"
done
if [[ "${STAGE}" == calibrate || "${STAGE}" == all ]]; then
  STATES="$(build_calib "${STATE_ARGV[@]}")"
else
  STATES="$(calib_path "${STATE_ARGV[@]}")"
fi
echo "[llada] held-out states: ${STATES}"

# Hard-fails unless clean and Traj-MC share byte-identical pre-noise windows.
"${PYTHON_BIN}" -m trajmc.analysis.calibration_ablation \
  --calib "clean_t0=${CALIB[clean_t0]}" --calib "random_t=${CALIB[random_t]}" \
  --out "${REPORT_DIR}/calibration_audit.json"

# ── 2. compression: same layers (all 7 linears per block), same rank ─────────
if [[ "${STAGE}" == compress || "${STAGE}" == all ]]; then
  for ratio in ${RATIOS}; do
    W="${OUT}/r${ratio}/weights"
    ARGS=("${BACKEND[@]}" --ratio "${ratio}" --layer_type all
          --linalg_device "${LINALG_DEVICE}" --batch_size 1 --save_dtype bfloat16)
    compress "${ARGS[@]}" --decomp identity \
      --save_path "${W}/weight_svd" --run_id "llada_r${ratio}_weight_svd"
    compress "${ARGS[@]}" --decomp cholesky --calib "${CALIB[clean_t0]}" \
      --save_path "${W}/clean" --run_id "llada_r${ratio}_clean"
    compress "${ARGS[@]}" --decomp cholesky --calib "${CALIB[random_t]}" \
      --save_path "${W}/traj" --run_id "llada_r${ratio}_traj"
  done
fi

# ── 3. E_gen on held-out generation states (Fig. 1b), 20% reduction ──────────
if [[ "${STAGE}" == measure || "${STAGE}" == all ]]; then
  W="${OUT}/r0.8/weights"
  for positions in generation all; do
    "${PYTHON_BIN}" -m trajmc.analysis.gen_reconstruction \
      "${BACKEND[@]}" --states "${STATES}" --layer_type all \
      --weights "weight_svd=${W}/weight_svd" --weights "clean=${W}/clean" \
      --weights "traj=${W}/traj" \
      --calib_manifest "$(manifest_for "${CALIB[clean_t0]}")" \
      --calib_manifest "$(manifest_for "${CALIB[random_t]}")" \
      --baseline clean --positions "${positions}" --bins "${RECON_BINS:-$(recon_bins)}" \
      --out "${REPORT_DIR}/gen_reconstruction_r0.8_${positions}.json"
  done
fi
echo "[llada] done -> ${OUT}"

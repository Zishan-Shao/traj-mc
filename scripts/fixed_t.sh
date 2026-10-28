#!/usr/bin/env bash
# Fixed-t control (Fig. 2a): is it the trajectory mixture, or any masking?
#
# Collapses pi(t) onto a single corruption level t in {0.25, 0.5, 0.75} over the
# identical C4 windows, budget, layers and rank allocation as the clean (t = 0)
# and Traj-MC (t ~ U(0,1)) arms, then measures E_gen on the same held-out
# generation states.  LLaDA-8B-Instruct, 20% reduction.
#
# Run scripts/llada_compress.sh first: the t = 0 and U(0,1) endpoints and the
# held-out states are reused from it, not rebuilt.
#
#   bash scripts/fixed_t.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE="${STAGE:-all}"
RATIO=0.8
FIXED_TS="${FIXED_TS:-0.25 0.5 0.75}"
LINALG_DEVICE="${LINALG_DEVICE:-cpu}"
EVAL_SEED="${EVAL_SEED:-1337}"
OUT="${RESULTS}/llada_instruct"
CALIB_DIR="${OUT}/calib"
STATE_DIR="${OUT}/heldout_states"
REPORT_DIR="${OUT}/reports"
W="${OUT}/r${RATIO}/weights"
mkdir -p "${REPORT_DIR}"

BACKEND=(--backend llada_instruct --model_path "${LLADA_PATH}")
BASE=("${BACKEND[@]}" "${CALIB_COMMON[@]}" --nsamples "${NSAMPLES}"
      --seed "${SEED}" --sampling_seed "${SEED}" --out_dir "${CALIB_DIR}")

require() { [[ -e "$1" ]] || { echo "missing $2: $1 (run scripts/llada_compress.sh)" >&2; exit 1; }; }
CLEAN="$(calib_path "${BASE[@]}" --scheme clean_t0)";  require "${CLEAN}" "clean calibration"
TRAJ="$(calib_path "${BASE[@]}" --scheme random_t)";   require "${TRAJ}" "Traj-MC calibration"
STATES="$(calib_path "${BACKEND[@]}" "${CALIB_COMMON[@]}" --scheme rollout \
  --rollout_prompts gsm8k_test --nsamples "${NSAMPLES}" --seed "${EVAL_SEED}" \
  --sampling_seed "${EVAL_SEED}" --rollout_steps 256 --rollout_gen_length 256 \
  --rollout_block_length 8 --out_dir "${STATE_DIR}")"
require "${STATES}" "held-out states"
require "${W}/clean/compression_summary.json" "clean weights"
require "${W}/traj/compression_summary.json" "Traj-MC weights"

# ── 1. calibration, one artifact per t ───────────────────────────────────────
LABELS=() ; declare -A FCALIB
for t in ${FIXED_TS}; do
  label="fixed_t${t//./p}"
  LABELS+=("${label}")
  if [[ "${STAGE}" == calibrate || "${STAGE}" == all ]]; then
    FCALIB[${label}]="$(build_calib "${BASE[@]}" --scheme fixed_t --fixed_t "${t}")"
  else
    FCALIB[${label}]="$(calib_path "${BASE[@]}" --scheme fixed_t --fixed_t "${t}")"
  fi
  echo "[fixed-t] ${label}: ${FCALIB[${label}]}"
done

# The single-variable claim rests on this: every arm's pre-noise windows must be
# byte-identical to the clean arm's.
AUDIT=(--calib "clean_t0=${CLEAN}" --calib "random_t=${TRAJ}")
for label in "${LABELS[@]}"; do AUDIT+=(--calib "${label}=${FCALIB[${label}]}"); done
"${PYTHON_BIN}" -m trajmc.analysis.calibration_ablation "${AUDIT[@]}" \
  --out "${REPORT_DIR}/fixed_t_calibration_audit.json"

# ── 2. compression ───────────────────────────────────────────────────────────
if [[ "${STAGE}" == compress || "${STAGE}" == all ]]; then
  for label in "${LABELS[@]}"; do
    compress "${BACKEND[@]}" --ratio "${RATIO}" --layer_type all \
      --decomp cholesky --calib "${FCALIB[${label}]}" \
      --linalg_device "${LINALG_DEVICE}" --batch_size 1 --save_dtype bfloat16 \
      --save_path "${W}/${label}" --run_id "llada_r${RATIO}_${label}"
  done
fi

# ── 3. E_gen ─────────────────────────────────────────────────────────────────
if [[ "${STAGE}" == measure || "${STAGE}" == all ]]; then
  RECON=(--weights "clean=${W}/clean")
  for label in "${LABELS[@]}"; do RECON+=(--weights "${label}=${W}/${label}"); done
  RECON+=(--weights "traj=${W}/traj"
          --calib_manifest "$(manifest_for "${CLEAN}")"
          --calib_manifest "$(manifest_for "${TRAJ}")")
  for label in "${LABELS[@]}"; do
    RECON+=(--calib_manifest "$(manifest_for "${FCALIB[${label}]}")")
  done
  for positions in generation all; do
    "${PYTHON_BIN}" -m trajmc.analysis.gen_reconstruction "${RECON[@]}" \
      "${BACKEND[@]}" --states "${STATES}" --layer_type all \
      --baseline clean --positions "${positions}" --bins "${RECON_BINS:-$(recon_bins)}" \
      --out "${REPORT_DIR}/fixed_t_r${RATIO}_${positions}.json"
  done
fi
echo "[fixed-t] reports -> ${REPORT_DIR}"

#!/usr/bin/env bash
# Stage A of the Traj-SVD study, on one model at one compression level.
#
# Two gates decide whether the full matrix is worth running:
#   RQ1  the Traj arm's kept low-rank subspace sits closer to the real
#        rollout subspace than the clean arm's;
#   RQ2  E_gen(Traj) < E_gen(Clean) on held-out real generation states.
# Nothing downstream should be launched until both are answered.
#
# Arms, all at the same rank allocation, layer scope, and calibration budget:
#   weight_svd  plain weight SVD, no activation statistics
#   clean       clean_t0        -- clean calibration text, t = 0
#   mcs         grid_t_prefix   -- released Quant-dLLM MCS (stratified grid,
#                                  25% visible prefix); byte-identical to
#                                  baselines/quant_dllm/utils/mcs.py
#   traj        random_t        -- Traj-SVD: iid t ~ U(0,1), full-window
#                                  forward-diffusion marginal
#   actual      rollout         -- control: real dense reverse-sampler states
#                                  on task prompts under the deployed block
#                                  sampler (see docs/TRAJ_SVD.md)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "${ROOT_DIR}/scripts/lib.sh"

BACKEND="${1:-llada_instruct}"
STAGE="${STAGE:-all}"
CORPUS="${CORPUS:-c4}"
NSAMPLES="${NSAMPLES:-256}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-42}"
SAMPLING_SEED="${SAMPLING_SEED:-${SEED}}"
EVAL_SEED="${EVAL_SEED:-1337}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-${NSAMPLES}}"
RATIO="${RATIO:-0.8}"
PREFIX_RATIO="${PREFIX_RATIO:-0.25}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
LAYER_TYPE="${LAYER_TYPE:-all}"
RECON_BINS="${RECON_BINS:-1}"
# Evaluation casts factors to bfloat16 anyway, so bfloat16 stores the deployed
# values without changing runtime math -- and halves the factor footprint.
SAVE_DTYPE="${SAVE_DTYPE:-float32}"
# Actual-State control: real deployment states, so task prompts and the
# official block-diffusion configuration.  LLaDA-8B-Instruct collapses into EOS
# padding without block diffusion (official EVAL.md: GSM8K 69.4 -> 78.6 at
# gen_length=256, block_length=8).
ROLLOUT_PROMPTS="${ROLLOUT_PROMPTS:-gsm8k_train}"
ROLLOUT_EVAL_PROMPTS="${ROLLOUT_EVAL_PROMPTS:-gsm8k_test}"
ROLLOUT_GEN_LENGTH="${ROLLOUT_GEN_LENGTH:-256}"
ROLLOUT_BLOCK_LENGTH="${ROLLOUT_BLOCK_LENGTH:-8}"
ROLLOUT_NUM_FEWSHOT="${ROLLOUT_NUM_FEWSHOT:-5}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/stage_a/${BACKEND}_r${RATIO}}"
# Calibration does not depend on the compression ratio, so a second ratio can
# point at the first one's tensors instead of recomputing them -- which is how
# the LLaDA r0.6 arms were built (same calib_sha256 as r0.8).
CALIB_DIR="${CALIB_DIR:-${OUT_ROOT}/calib}"
STATE_DIR="${OUT_ROOT}/heldout_states"
WEIGHT_DIR="${OUT_ROOT}/weights"
REPORT_DIR="${OUT_ROOT}/reports"

case "${STAGE}" in
  calibrate|compress|measure|all) ;;
  *) echo "STAGE must be calibrate, compress, measure, or all; got ${STAGE}" >&2
     exit 2 ;;
esac
if [[ "${EVAL_SEED}" == "${SEED}" ]]; then
  echo "EVAL_SEED must differ from SEED so E_gen uses held-out prompts" >&2
  exit 2
fi

# The full set, and the subset actually run.  A backend whose Actual-State
# rollout is not implemented yet (Dream: trajmc.calibration routes task-prompt
# rollouts through llada_block_rollout_states) can still produce every other
# arm by naming them, e.g.
#   SCHEMES=clean_t0,random_t,grid_t_prefix ARMS=weight_svd,clean,mcs,traj
# Held-out states and the two gates are skipped when `rollout` is not in the
# scheme list, because both are defined against real sampler states.
IFS=',' read -r -a CALIB_SCHEMES <<< "${SCHEMES:-clean_t0,random_t,grid_t_prefix,rollout}"
IFS=',' read -r -a ARM_LABELS <<< "${ARMS:-weight_svd,clean,mcs,traj,actual}"
WANT_ROLLOUT=0
for _s in "${CALIB_SCHEMES[@]}"; do [[ "${_s}" == rollout ]] && WANT_ROLLOUT=1; done
if [[ "${WANT_ROLLOUT}" == 0 ]]; then
  for _a in "${ARM_LABELS[@]}"; do
    [[ "${_a}" == actual ]] && { echo "ARMS includes 'actual' but SCHEMES has no 'rollout'" >&2; exit 2; }
  done
  if [[ "${STAGE}" == "measure" || "${STAGE}" == "all" ]]; then
    echo "[stage-a] no rollout scheme: skipping the two gates (they need real sampler states)"
    [[ "${STAGE}" == "measure" ]] && exit 0
  fi
fi
declare -A ARM_SCHEME=(
  [clean]=clean_t0
  [mcs]=grid_t_prefix
  [traj]=random_t
  [actual]=rollout
)

cd "${ROOT_DIR}"

COMMON=(
  --backend "${BACKEND}" --corpus "${CORPUS}" --seqlen "${SEQLEN}"
  --prefix_ratio "${PREFIX_RATIO}" --rollout_steps "${ROLLOUT_STEPS}"
)
if [[ -n "${MODEL_PATH:-}" ]]; then
  COMMON+=(--model_path "${MODEL_PATH}")
fi
if [[ "${CORPUS}" == "c4" && "${C4_STREAMING:-1}" == "1" ]]; then
  COMMON+=(--c4_streaming)
fi

# Sets CALIB_ARGV to the argv used both to resolve a path and to run the job,
# so the two can never describe different files.
build_calib_argv() {  # scheme nsamples seed sampling_seed out_dir [prompts]
  CALIB_ARGV=(
    "${COMMON[@]}" --scheme "$1" --nsamples "$2" --seed "$3"
    --sampling_seed "$4" --out_dir "$5"
  )
  if [[ "$1" == "rollout" ]]; then
    CALIB_ARGV+=(
      --rollout_prompts "${6:-${ROLLOUT_PROMPTS}}"
      --rollout_gen_length "${ROLLOUT_GEN_LENGTH}"
      --rollout_block_length "${ROLLOUT_BLOCK_LENGTH}"
      --rollout_num_fewshot "${ROLLOUT_NUM_FEWSHOT}"
    )
  fi
}

# ── preflight ─────────────────────────────────────────────────────────────────
# Every artifact path is resolved by trajmc.calibration itself before any GPU
# work starts.  A bad backend, seed, prefix ratio, or step count fails here, in
# seconds, rather than after the two dense rollouts have been paid for.
declare -A CALIB_PATH
declare -A CALIB_ARGV_OF
for scheme in "${CALIB_SCHEMES[@]}"; do
  build_calib_argv "${scheme}" "${NSAMPLES}" "${SEED}" "${SAMPLING_SEED}" "${CALIB_DIR}"
  CALIB_PATH["${scheme}"]="$(calib_artifact_path "${CALIB_ARGV[@]}")"
done
EVAL_STATES=""
if [[ "${WANT_ROLLOUT}" == 1 ]]; then
  build_calib_argv rollout "${EVAL_NSAMPLES}" "${EVAL_SEED}" "${EVAL_SEED}" \
    "${STATE_DIR}" "${ROLLOUT_EVAL_PROMPTS}"
  EVAL_STATES="$(calib_artifact_path "${CALIB_ARGV[@]}")"
fi

echo "[stage-a] backend=${BACKEND} ratio=${RATIO} stage=${STAGE}"
if [[ "${WANT_ROLLOUT}" == 1 ]]; then
  echo "[stage-a] actual-state: ${ROLLOUT_PROMPTS} (calib) / ${ROLLOUT_EVAL_PROMPTS} (eval)"
else
  echo "[stage-a] actual-state: not in this run (SCHEMES=${SCHEMES:-<default>})"
fi
echo "[stage-a]   gen_length=${ROLLOUT_GEN_LENGTH} block_length=${ROLLOUT_BLOCK_LENGTH} steps=${ROLLOUT_STEPS}"
for scheme in "${CALIB_SCHEMES[@]}"; do
  echo "[stage-a]   ${scheme} -> ${CALIB_PATH[${scheme}]}"
done
echo "[stage-a]   heldout states -> ${EVAL_STATES}"

# Later stages consume artifacts the calibrate stage wrote; check them up front.
if [[ "${STAGE}" != "calibrate" && "${STAGE}" != "all" ]]; then
  for scheme in "${CALIB_SCHEMES[@]}"; do
    require_calib_artifact "${CALIB_PATH[${scheme}]}" "${scheme}"
  done
  [[ "${WANT_ROLLOUT}" == 1 ]] && require_calib_artifact "${EVAL_STATES}" "held-out states"
fi

# ── 1. calibration states (shared clean windows) + held-out evaluation states ──
if [[ "${STAGE}" == "calibrate" || "${STAGE}" == "all" ]]; then
  for scheme in "${CALIB_SCHEMES[@]}"; do
    build_calib_argv "${scheme}" "${NSAMPLES}" "${SEED}" "${SAMPLING_SEED}" \
      "${CALIB_DIR}"
    "${PYTHON_BIN}" -m trajmc.calibration "${CALIB_ARGV[@]}"
    require_calib_artifact "${CALIB_PATH[${scheme}]}" "${scheme}"
  done
  # E_gen is measured on real sampler states from prompts no arm calibrated on.
  if [[ "${WANT_ROLLOUT}" == 1 ]]; then
    build_calib_argv rollout "${EVAL_NSAMPLES}" "${EVAL_SEED}" "${EVAL_SEED}" \
      "${STATE_DIR}" "${ROLLOUT_EVAL_PROMPTS}"
    "${PYTHON_BIN}" -m trajmc.calibration "${CALIB_ARGV[@]}"
    require_calib_artifact "${EVAL_STATES}" "held-out states"
  fi
fi

mkdir -p "${REPORT_DIR}"
AUDIT_ARGS=()
for scheme in "${CALIB_SCHEMES[@]}"; do
  AUDIT_ARGS+=(--calib "${scheme}=${CALIB_PATH[${scheme}]}")
done
# Fails unless every arm shares byte-identical pre-noise windows.
"${PYTHON_BIN}" -m analysis.calibration_ablation "${AUDIT_ARGS[@]}" \
  --out "${REPORT_DIR}/calibration_audit.json"

# ── 2. compression: one arm per calibration distribution, matched everywhere ───
if [[ "${STAGE}" == "compress" || "${STAGE}" == "all" ]]; then
  for label in "${ARM_LABELS[@]}"; do
    ARGS=(
      --backend "${BACKEND}" --ratio "${RATIO}" --layer_type "${LAYER_TYPE}"
      --save_path "${WEIGHT_DIR}/${label}" --run_id "stage_a_${label}"
      --linalg_device "${LINALG_DEVICE:-cpu}" --batch_size "${BATCH_SIZE:-1}"
      --save_dtype "${SAVE_DTYPE}"
    )
    if [[ "${label}" == "weight_svd" ]]; then
      ARGS+=(--decomp identity)
    else
      ARGS+=(--calib "${CALIB_PATH[${ARM_SCHEME[${label}]}]}"
             --decomp "${DECOMP:-cholesky}")
    fi
    if [[ -n "${MODEL_PATH:-}" ]]; then ARGS+=(--model_path "${MODEL_PATH}"); fi
    if [[ -n "${XTX_BUDGET_GB:-}" ]]; then
      ARGS+=(--xtx_budget_gb "${XTX_BUDGET_GB}")
    fi
    echo "[stage-a] compress ${label}"
    "${PYTHON_BIN}" -m trajmc.compression "${ARGS[@]}"
  done
fi

# ── 3. the two gates ──────────────────────────────────────────────────────────
if [[ "${WANT_ROLLOUT}" == 1 && ( "${STAGE}" == "measure" || "${STAGE}" == "all" ) ]]; then
  SUBSPACE_ARGS=()
  for scheme in "${CALIB_SCHEMES[@]}"; do
    SUBSPACE_ARGS+=(--calib "${scheme}=${CALIB_PATH[${scheme}]}")
  done
  # RQ1: which subspace does each calibration keep, relative to the real one?
  "${PYTHON_BIN}" -m analysis.subspace_distance "${SUBSPACE_ARGS[@]}" \
    --backend "${BACKEND}" --reference rollout --ratio "${RATIO}" \
    ${MODEL_PATH:+--model_path "${MODEL_PATH}"} \
    ${SUBSPACE_SUFFIX:+--suffix "${SUBSPACE_SUFFIX}"} \
    --block_stride "${BLOCK_STRIDE:-1}" \
    --out "${REPORT_DIR}/subspace_distance_r${RATIO}.json"

  RECON_ARGS=()
  for label in "${ARM_LABELS[@]}"; do
    RECON_ARGS+=(--weights "${label}=${WEIGHT_DIR}/${label}")
  done
  for scheme in "${CALIB_SCHEMES[@]}"; do
    RECON_ARGS+=(--calib_manifest "$(manifest_for "${CALIB_PATH[${scheme}]}")")
  done
  # RQ2: output reconstruction error on held-out real generation states.
  "${PYTHON_BIN}" -m analysis.gen_reconstruction "${RECON_ARGS[@]}" \
    --backend "${BACKEND}" --states "${EVAL_STATES}" \
    --baseline clean --oracle actual --layer_type "${LAYER_TYPE}" \
    --bins "${RECON_BINS}" \
    ${MODEL_PATH:+--model_path "${MODEL_PATH}"} \
    --out "${REPORT_DIR}/gen_reconstruction_r${RATIO}.json"
fi

echo "[stage-a] reports -> ${REPORT_DIR}"

#!/usr/bin/env bash
# Monte Carlo convergence of the trajectory second moment (Fig. 2b).
#
# Builds Traj-MC calibrations at N in {16, 32, 64, 128, 256} windows (window
# seed 42; budget N is the first N windows and the first N timesteps of the same
# stream) and an independent full-budget reference with window seed 31337, then
# reports the relative Frobenius error of Sigma_N against that reference on the
# three probe layers (blocks.0.attn_out, blocks.16.attn_out, blocks.31.up_proj).
# The clean (t = 0) N = 256 moment rides along as the population-mismatch
# control.  LLaDA-8B-Instruct; no compression is needed for this curve.
#
# Run scripts/llada_compress.sh first (the N = 256 artifacts are reused).
#
#   bash scripts/mc_budget.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUDGETS="${BUDGETS:-16 32 64 128 256}"
REF_SEED="${REF_SEED:-31337}"
OUT="${RESULTS}/llada_instruct"
CALIB_DIR="${OUT}/calib"            # N = 256 lives here (shared with llada_compress.sh)
BUDGET_DIR="${OUT}/calib_budget"    # N < 256
REF_DIR="${OUT}/calib_reference"    # independent draw
REPORT_DIR="${OUT}/reports"
mkdir -p "${BUDGET_DIR}" "${REF_DIR}" "${REPORT_DIR}"

BACKEND=(--backend llada_instruct --model_path "${LLADA_PATH}")
argv() {  # scheme nsamples seed out_dir
  ARGV=("${BACKEND[@]}" "${CALIB_COMMON[@]}" --scheme "$1" --nsamples "$2"
        --seed "$3" --sampling_seed "$3" --out_dir "$4")
}

COV=()
MANIFESTS=()
for n in ${BUDGETS}; do
  dir="${BUDGET_DIR}"; [[ "${n}" == 256 ]] && dir="${CALIB_DIR}"
  argv random_t "${n}" "${SEED}" "${dir}"
  path="$(build_calib "${ARGV[@]}")"
  COV+=(--calib "random_t_n${n}=${path}")
  MANIFESTS+=("$(manifest_for "${path}")")
done
argv clean_t0 256 "${SEED}" "${CALIB_DIR}"
COV+=(--calib "clean_t0_n256=$(build_calib "${ARGV[@]}")")
argv random_t 256 "${REF_SEED}" "${REF_DIR}"
COV+=(--calib "independent_random_t=$(build_calib "${ARGV[@]}")")

# Windows and timesteps must nest across budgets (mask realisations do not, by
# construction); a broken input would otherwise read as "does not converge".
"${PYTHON_BIN}" - "${MANIFESTS[@]}" <<'PY'
import json, sys
ms = sorted((json.load(open(p)) for p in sys.argv[1:]), key=lambda m: m["nsamples"])
for field in ("window_hashes_prenoise", "per_sample_t"):
    for a, b in zip(ms, ms[1:]):
        if b[field][: len(a[field])] != a[field]:
            raise SystemExit(f"N={a['nsamples']} {field} is not a prefix of N={b['nsamples']}")
print("[mc-budget] windows and timesteps nest across N =", [m["nsamples"] for m in ms])
PY

"${PYTHON_BIN}" -m trajmc.analysis.covariance_estimation "${COV[@]}" \
  "${BACKEND[@]}" --reference independent_random_t \
  --out "${REPORT_DIR}/mc_budget_covariance.json"
echo "[mc-budget] report -> ${REPORT_DIR}/mc_budget_covariance.json"

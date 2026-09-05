#!/usr/bin/env bash
# MATH-500 protocol pilot: which official LLaDA-8B-Instruct sampling family
# does the column belong in?  LLaDA publishes MATH twice --
#
#   pure diffusion   gen 512 / block 512, confidence_eos_eot_inf=True   -> 29.6
#   block diffusion  gen 512 / block  64, both eos flags False          -> 42.7
#
# -- and the finished column ran the first.  (gen 256 / block 256 is *not* an
# Instruct setting; it appears only in the Base model's lm-eval sweep, so it is
# not tested here.)
#
# Only the block-diffusion side needs GPUs: the pure-diffusion side is the same
# 100 problems sliced out of the finished full runs, at zero cost.  Three jobs.
#
#   scripts/run_math500_pilot.sh          # submit whatever is missing
#   DRY=1 scripts/run_math500_pilot.sh
#
# Read-out is analysis/math500_pilot_report.py, which applies the rule
# pre-registered in results/pilot_math500/items_manifest.json: keep the
# finished column if Traj > Clean holds in both families with the same sign;
# rerun it under block diffusion only if the ordering flips or the gap changes
# sign.  A higher block-diffusion score is explicitly not a reason to switch.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PARTITION="${PARTITION:-athena-genai,athena-small,athena,athena-mini}"
STAGE_ROOT="${STAGE_ROOT:-results/stage_a}"
MODEL="${MODEL:-llada_instruct}"
DRY="${DRY:-0}"

declare -A CELL_WEIGHTS=(
  [dense]=dense
  [r08_clean]="${STAGE_ROOT}/${MODEL}_r0.8/weights/clean"
  [r08_traj]="${STAGE_ROOT}/${MODEL}_r0.8/weights/traj"
)
# cfg tag -> gen:block:steps:logits_eos_inf:confidence_eos_eot_inf.  Only the
# block-diffusion arm is submitted; "cur" stays documented here because the
# read-out slices it out of the finished runs rather than rerunning it.
declare -A CFGS=([blk]=512:64:512:0:0)

for cell in dense r08_clean r08_traj; do
  weights="${CELL_WEIGHTS[$cell]}"
  if [[ "${weights}" != dense && ! -f "${weights}/compression_summary.json" ]]; then
    echo "missing weights for ${cell}: ${weights}" >&2; exit 2
  fi
  for cfg in "${!CFGS[@]}"; do
    IFS=: read -r gen blk stp lei ceei <<< "${CFGS[$cfg]}"
    if [[ -f "results/pilot_math500/${cell}_${cfg}.json" ]]; then
      echo "[done]   ${cell} ${cfg}"; continue
    fi
    if squeue -u "${USER}" -h -o '%j' | grep -qxF "tP-${cell}-${cfg}"; then
      echo "[queued] ${cell} ${cfg}"; continue
    fi
    echo "[submit] ${cell} ${cfg}  gen=${gen} block=${blk} steps=${stp} L=${lei} C=${ceei}"
    [[ "${DRY}" == 1 ]] && continue
    sbatch --partition="${PARTITION}" --job-name="tP-${cell}-${cfg}" \
      --export=ALL,CELL="${cell}",WEIGHTS="${weights}",CFG="${cfg}",GEN="${gen}",BLK="${blk}",STP="${stp}",LEI="${lei}",CEEI="${ceei}" \
      scripts/slurm/math500_pilot.sbatch
  done
done

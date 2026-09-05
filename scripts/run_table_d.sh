#!/usr/bin/env bash
# Fill the HumanEval / MBPP / IFEval / BBH columns of the LLaDA-Instruct
# low-rank table on the checkpoints that already carry its GSM8K / MATH-500 /
# MMLU numbers.  No new arms: one job per (existing cell x benchmark).  All
# four columns are opt-in now -- see BENCHES below for why.
#
#   BM=ifeval scripts/run_table_d.sh    # submit everything still missing
#   BM=bbh scripts/run_table_d.sh       # every column here is opt-in, see BENCHES below
#   CELLS=dense,r08_traj scripts/run_table_d.sh
#   DRY=1 scripts/run_table_d.sh        # print the plan, submit nothing
#
# Long jobs stream their raw generations to results/eval_stageB/<cell>/gens/,
# keyed by question, so a killed or preempted job resumes instead of starting
# the benchmark over -- and a later, larger BBH_PER_SUBTASK reuses every answer
# already produced.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Every partition here is PriorityTier=1, so a multi-partition job simply
# starts wherever a card frees up first; athena-genai leads because node5/6 are
# L40S and the rest are A5000s, measured at ~1/3 the throughput (116 s vs 39 s
# per HumanEval question) -- worth taking when they would otherwise sit idle.
PARTITION="${PARTITION:-athena-genai,athena-small,athena,athena-mini}"
BBH_PER_SUBTASK="${BBH_PER_SUBTASK:-40}"
STAGE_ROOT="${STAGE_ROOT:-results/stage_a}"
MODEL="${MODEL:-llada_instruct}"
DRY="${DRY:-0}"

# The nine cells these columns cover.  MCS is deliberately absent: r06_mcs has
# no row in the table at all, and the r08_mcs arm was dropped from these
# columns on request, so neither is submitted.  Both keep their existing
# GSM8K / MATH-500 / MMLU results -- only the four new columns skip them.
CELL_ORDER=(dense
            r08_weight_svd r08_clean r08_traj r08_actual
            r06_weight_svd r06_clean r06_traj r06_actual)
declare -A CELL_WEIGHTS=(
  [dense]=dense
  [r08_weight_svd]="${STAGE_ROOT}/${MODEL}_r0.8/weights/weight_svd"
  [r08_clean]="${STAGE_ROOT}/${MODEL}_r0.8/weights/clean"
  [r08_traj]="${STAGE_ROOT}/${MODEL}_r0.8/weights/traj"
  [r08_actual]="${STAGE_ROOT}/${MODEL}_r0.8/weights/actual"
  [r06_weight_svd]="${STAGE_ROOT}/${MODEL}_r0.6/weights/weight_svd"
  [r06_clean]="${STAGE_ROOT}/${MODEL}_r0.6/weights/clean"
  [r06_traj]="${STAGE_ROOT}/${MODEL}_r0.6/weights/traj"
  [r06_actual]="${STAGE_ROOT}/${MODEL}_r0.6/weights/actual"
  [r08_mcs]="${STAGE_ROOT}/${MODEL}_r0.8/weights/mcs"
)

# Nothing is submitted by default any more: all four columns this script drives
# have been called off.  HumanEval and MBPP went first on 2026-09-03 (every
# compressed arm sat at the floor, 0-12%, so the columns could not separate
# calibration schemes), BBH before them (~12 h per arm even at 40 questions per
# subtask), and IFEval later the same day.  Finished results and generation
# caches all stay on disk, so asking for one explicitly -- BM=bbh
# scripts/run_table_d.sh -- resumes rather than restarts.  The columns that are
# still live (svamp / arc_c / arc_e / hellaswag / piqa / aime / minerva_math)
# go through scripts/run_eval_workers.sh instead, which holds GPUs across a
# whole queue rather than submitting one job per cell.
if [[ -z "${BM:-}" ]]; then
  echo "set BM=<humaneval|mbpp|ifeval|bbh>; these columns are no longer run by default" >&2
  exit 2
fi
IFS=',' read -r -a BENCHES <<< "${BM}"

# Roughly twice the measured single-L40S cost per arm (39 s/question at
# gen_length 512 on HumanEval, 30 s on MBPP, 42 s on IFEval, ~35 s on BBH).
# A wall limit is worth setting even though athena-genai imposes none: the
# backfill scheduler will only slot a job into a reservation window it fits
# inside, and a job that does hit the limit resumes from its generation cache.
# Sized for the slowest card a job can land on (A5000: 116 s/question on
# HumanEval, ~85 s on MBPP, 103 s on IFEval), with headroom.  A job that does
# hit the limit resumes from its generation cache when resubmitted.
declare -A TIME_LIMIT=(
  [humaneval]=08:00:00 [mbpp]=18:00:00 [ifeval]=22:00:00 [bbh]=40:00:00
  [aime]=06:00:00 [minerva_math]=24:00:00 [svamp]=48:00:00
)
IFS=',' read -r -a CELLS <<< "${CELLS:-$(IFS=,; echo "${CELL_ORDER[*]}")}"

# Cells already queued or running have no result file yet, so the check below
# would resubmit them.  One squeue call up front instead.
QUEUED="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"

n_submit=0 n_skip=0
for bm in "${BENCHES[@]}"; do
  for cell in "${CELLS[@]}"; do
    weights="${CELL_WEIGHTS[${cell}]:-}"
    if [[ -z "${weights}" ]]; then
      echo "unknown cell ${cell}; known: ${CELL_ORDER[*]}" >&2
      exit 2
    fi
    if [[ "${weights}" != "dense" && ! -f "${weights}/compression_summary.json" ]]; then
      echo "missing compressed weights for ${cell}: ${weights}" >&2
      exit 2
    fi
    if [[ -f "results/eval_stageB/${cell}/${bm}.json" ]]; then
      n_skip=$((n_skip + 1))
      continue
    fi
    if grep -qxF "tD-${cell}-${bm}" <<< "${QUEUED}"; then
      echo "[queued] ${cell} ${bm}"
      n_skip=$((n_skip + 1))
      continue
    fi
    echo "[submit] ${cell} ${bm}"
    if [[ "${DRY}" == "1" ]]; then
      n_submit=$((n_submit + 1))
      continue
    fi
    sbatch --partition="${PARTITION}" --job-name="tD-${cell}-${bm}" \
      --time="${TIME_LIMIT[${bm}]:-24:00:00}" \
      --export=ALL,CELL="${cell}",WEIGHTS="${weights}",BM="${bm}",PARTITION="${PARTITION//,/+}",BBH_PER_SUBTASK="${BBH_PER_SUBTASK}" \
      scripts/slurm/table_d.sbatch
    n_submit=$((n_submit + 1))
  done
done

echo "[run_table_d] submitted ${n_submit}, already done ${n_skip}  (partition=${PARTITION}, BBH cap ${BBH_PER_SUBTASK}/subtask)"

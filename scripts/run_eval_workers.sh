#!/usr/bin/env bash
# Queue the LLaDA-Instruct low-rank table's remaining columns and start GPU
# workers that hold their cards until the queue is empty
# (scripts/slurm/eval_worker.sbatch).
#
#   scripts/run_eval_workers.sh                 # queue everything missing, start workers
#   N_NODE5=4 N_NODE6=4 N_ANY=8 scripts/run_eval_workers.sh
#   BM=aime CELLS=dense,r08_traj DRY=1 scripts/run_eval_workers.sh
#
# The queue is column-major and ordered cheapest column first -- the four
# zero-shot multiple-choice columns answer with a single option letter (3
# generated tokens, minutes per arm) while SVAMP and Minerva Math run a full
# 256/512-step CoT (hours per arm) -- so whole columns complete early rather
# than one cell finishing every benchmark while the others have none.
# Re-running only appends tasks that are neither done nor already queued;
# workers already running pick the new lines up on their next pass.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# N_NODE5 / N_NODE6 workers are pinned to node5 / node6 (L40S, ~2.6x an A5000)
# and wait for a card there; N_ANY workers take the first card that frees
# anywhere.  Pinned workers are worth queueing past the eight cards a node has:
# each one claims an L40S the moment somebody else's job releases it.
N_NODE5="${N_NODE5:-4}"
N_NODE6="${N_NODE6:-4}"
N_ANY="${N_ANY:-4}"
ANY_PARTITION="${ANY_PARTITION:-athena-genai,athena-small,athena,athena-mini}"
# Slurm backfills a worker onto a card that is reserved for a job further up
# the queue only if the worker ends before that reservation starts, so the time
# limit -- not the queue position -- is what decides whether a worker waits a
# day or slots into a gap this afternoon.  node5 and node6 carry their own
# value because a reservation on one does not shadow the other: a node5 8-GPU
# reservation caps what node5 can accept while node6 still takes a long job.
WORKER_TIME="${WORKER_TIME:-7-00:00:00}"
TIME_NODE5="${TIME_NODE5:-${WORKER_TIME}}"
TIME_NODE6="${TIME_NODE6:-${WORKER_TIME}}"
TIME_ANY="${TIME_ANY:-${WORKER_TIME}}"
STAGE_ROOT="${STAGE_ROOT:-results/stage_a}"
MODEL="${MODEL:-llada_instruct}"
QUEUE_DIR="results/eval_stageB/.queue"
QUEUE_FILE="${QUEUE_DIR}/tasks.txt"
DRY="${DRY:-0}"
mkdir -p "${QUEUE_DIR}/claims" results/slurm

# Cheapest column first: piqa / arc_c / arc_e / hellaswag are single-letter
# answers (~5-45 min per arm), aime is 60 problems, then the long CoT columns.
IFS=',' read -r -a BENCHES <<< "${BM:-piqa,arc_c,arc_e,hellaswag,aime,minerva_math,svamp}"
# The ten cells of the plan's LLaDA-Instruct low-rank table: dense plus five
# arms at 20% reduction and four at 40%.  r06_mcs is compressed and on disk but
# has no row in that table (MCS-SVD is listed only at 20%), so it is reachable
# through CELLS=r06_mcs rather than run by default.
CELL_ORDER=(dense r08_weight_svd r08_clean r08_mcs r08_traj r08_actual
            r06_weight_svd r06_clean r06_traj r06_actual)
IFS=',' read -r -a CELLS <<< "${CELLS:-$(IFS=,; echo "${CELL_ORDER[*]}")}"

n_new=0 n_skip=0
touch "${QUEUE_FILE}"
for bm in "${BENCHES[@]}"; do
  for cell in "${CELLS[@]}"; do
    if [[ "${cell}" != dense ]]; then
      w="${STAGE_ROOT}/${MODEL}_r0.${cell:2:1}/weights/${cell#r0?_}"
      [[ -f "${w}/compression_summary.json" ]] || { echo "missing weights for ${cell}: ${w}" >&2; exit 2; }
    fi
    if [[ -f "results/eval_stageB/${cell}/${bm}.json" ]] || grep -qx "${cell}:${bm}" "${QUEUE_FILE}"; then
      n_skip=$((n_skip + 1)); continue
    fi
    echo "[queue] ${cell} ${bm}"
    [[ "${DRY}" == 1 ]] || echo "${cell}:${bm}" >> "${QUEUE_FILE}"
    n_new=$((n_new + 1))
  done
done
echo "[run_eval_workers] queued ${n_new} new, ${n_skip} done/already queued -> ${QUEUE_FILE}"

running="$(squeue -u "${USER}" -h -o '%j' | grep -c '^tM-worker' || true)"
echo "[run_eval_workers] workers already submitted: ${running}"
if [[ "${DRY}" == 1 ]]; then exit 0; fi
for i in $(seq 1 "${N_NODE5}"); do
  sbatch --partition=athena-genai --nodelist=node5 --time="${TIME_NODE5}" \
    --job-name=tM-worker-node5 --export=ALL,STAGE_ROOT="${STAGE_ROOT}",MODEL="${MODEL}" \
    scripts/slurm/eval_worker.sbatch
done
for i in $(seq 1 "${N_NODE6}"); do
  sbatch --partition=athena-genai --nodelist=node6 --time="${TIME_NODE6}" \
    --job-name=tM-worker-node6 --export=ALL,STAGE_ROOT="${STAGE_ROOT}",MODEL="${MODEL}" \
    scripts/slurm/eval_worker.sbatch
done
for i in $(seq 1 "${N_ANY}"); do
  sbatch --partition="${ANY_PARTITION}" --time="${TIME_ANY}" \
    --job-name=tM-worker --export=ALL,STAGE_ROOT="${STAGE_ROOT}",MODEL="${MODEL}" \
    scripts/slurm/eval_worker.sbatch
done
echo "[run_eval_workers] submitted ${N_NODE5} node5-pinned + ${N_NODE6} node6-pinned + ${N_ANY} floating workers"

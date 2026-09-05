#!/usr/bin/env bash
# One GPU that works through a queue of (cell x benchmark) evaluations and
# only gives the card back when nothing is left.  Submitted by
# scripts/run_eval_workers.sh; several workers share one queue file.
#
# Why a worker instead of one job per cell: every time a per-cell job ends the
# card goes back to the scheduler and the next job in line -- usually someone
# else's -- takes it.  A worker holds its allocation across the whole list, so
# a card we have stays ours until our list is empty.  It never idles: with the
# queue drained it exits at once.
#
# Claiming is an atomic mkdir under QUEUE_DIR/claims; the claim records the
# SLURM job id so a claim whose job is no longer running is treated as stale
# and re-taken (the generation cache means the retry only pays for what the
# dead job had not written).
#SBATCH --job-name=tM-worker
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --output=results/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/tl356/miniconda3/envs/AKS/bin/python}"
MODEL_PATH="${MODEL_PATH:-/zpool-00/home/tl356/LLaDA-8B-Instruct}"
STAGE_ROOT="${STAGE_ROOT:-results/stage_a}"
MODEL="${MODEL:-llada_instruct}"
QUEUE_DIR="${QUEUE_DIR:-results/eval_stageB/.queue}"
QUEUE_FILE="${QUEUE_FILE:-${QUEUE_DIR}/tasks.txt}"
mkdir -p "${QUEUE_DIR}/claims"

# HOLD=1 keeps the allocation for the whole --time instead of handing the card
# back.  The three paths below all end in "give the card to the scheduler", and
# on this cluster the scheduler gives it to whoever is next in line, not back to
# us.  When the card is wanted more than the work is, park here instead.
HOLD="${HOLD:-0}"
hold_or_exit() {                      # $1 = why
  if [[ "${HOLD}" == 1 ]]; then
    echo "[W] ${1}: HOLD=1, keeping GPU ${GPU_IDX:-?} on $(hostname) until the time limit"
    while true; do sleep 300; done    # only Slurm's TimeLimit ends this
  fi
  exit 0
}

# Same guard as table_d.sbatch: a card that fails a trivial matmul (the ECC
# card in this pool) is held here so it leaves the pool; the work is not lost
# because another worker will claim it.
if ! "$PYTHON_BIN" -c "import torch; x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite((x@x).sum())" 2>/dev/null; then
  echo "[W] GPU self-test failed on $(hostname); holding this card out of the pool"
  # Unlike the contended path this did not resubmit, so a bad card silently cost
  # a worker.  Ask for another card before parking on this one.
  sbatch --partition="${SLURM_JOB_PARTITION}" --job-name="${SLURM_JOB_NAME}" \
    --time="${SLURM_TIMELIMIT:-24:00:00}" --export=ALL,STAGE_ROOT="${STAGE_ROOT}",MODEL="${MODEL}",HOLD="${HOLD}" \
    "${BASH_SOURCE[0]}" || true
  sleep "${QUARANTINE_SECONDS:-86400}"
  exit 0
fi
# Slurm gives this job one GPU but cannot stop a process that ignores its own
# allocation from sitting on it too.  A shared card ran 2.7x slower in
# measurement (244 s/question against 89 s alone), which shows up as "this arm
# is slow" rather than as an error, so check it here instead.
GPU_IDX="${CUDA_VISIBLE_DEVICES%%,*}"
FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
          -i "${GPU_IDX}" 2>/dev/null | wc -l)
if [ "${FOREIGN:-0}" -gt 0 ]; then
  echo "[W] GPU ${GPU_IDX} on $(hostname) already has ${FOREIGN} process(es) on it; handing it back"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${GPU_IDX}" || true
  sbatch --partition="${SLURM_JOB_PARTITION}" --job-name="${SLURM_JOB_NAME}" \
    --export=ALL,STAGE_ROOT="${STAGE_ROOT}",MODEL="${MODEL}" "${BASH_SOURCE[0]}" || true
  sleep "${CONTENDED_HOLD_SECONDS:-600}"
  hold_or_exit "card contended"
fi

echo "[W] worker ${SLURM_JOB_ID} up on $(hostname) GPU ${GPU_IDX} $(date -Is)"

weights_for() {                       # cell -> weights dir (or "dense")
  case "$1" in
    dense) echo dense ;;
    r08_*) echo "${STAGE_ROOT}/${MODEL}_r0.8/weights/${1#r08_}" ;;
    r06_*) echo "${STAGE_ROOT}/${MODEL}_r0.6/weights/${1#r06_}" ;;
    *) return 1 ;;
  esac
}

claim() {                             # cell bench -> 0 if this worker now owns it
  local dir="${QUEUE_DIR}/claims/$1__$2"
  if mkdir "$dir" 2>/dev/null; then echo "${SLURM_JOB_ID}" > "$dir/owner"; return 0; fi
  local owner; owner="$(cat "$dir/owner" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && squeue -h -j "$owner" -o %i 2>/dev/null | grep -qx "$owner"; then
    return 1                          # someone alive holds it
  fi
  if [[ -f "$dir/failed" ]]; then return 1; fi   # needs a human look, do not loop on it
  echo "${SLURM_JOB_ID}" > "$dir/owner"          # stale claim: take it over
  echo "[W] re-taking stale claim $1/$2 (was job ${owner:-?})"
  return 0
}

n_done=0 idle=0
while true; do
  picked=""
  while IFS=: read -r cell bm; do
    [[ -z "${cell}" || "${cell}" == \#* ]] && continue
    out="results/eval_stageB/${cell}"
    [[ -f "${out}/${bm}.json" ]] && continue
    if claim "${cell}" "${bm}"; then picked="${cell}:${bm}"; break; fi
  done < "${QUEUE_FILE}"
  # Nothing claimable. That is not the same as "nothing left to do": every
  # remaining line may be held by a peer that is about to die or be killed, and
  # a queue file being rewritten is momentarily all-claimed too. Releasing the
  # card here loses it to the scheduler for good, so wait and re-check while
  # unfinished work still exists; only a genuinely finished queue exits at once.
  if [[ -z "${picked}" ]]; then
    left=0
    while IFS=: read -r cell bm; do
      [[ -z "${cell}" || "${cell}" == \#* ]] && continue
      [[ -f "results/eval_stageB/${cell}/${bm}.json" ]] || left=$((left + 1))
    done < "${QUEUE_FILE}"
    if [[ ${left} -eq 0 ]]; then break; fi
    idle=$((idle + 1))
    if [[ ${idle} -gt ${IDLE_ROUNDS:-60} ]]; then
      echo "[W] ${left} task(s) unfinished but all claimed for $((idle - 1)) min; leaving the task loop"
      break
    fi
    [[ ${idle} -eq 1 ]] && echo "[W] all ${left} unfinished task(s) claimed; holding card and re-checking"
    sleep 60
    continue
  fi
  idle=0

  cell="${picked%%:*}"; bm="${picked##*:}"
  out="results/eval_stageB/${cell}"; mkdir -p "${out}"
  weights="$(weights_for "${cell}")" || { echo "[W] unknown cell ${cell}" >&2; touch "${QUEUE_DIR}/claims/${cell}__${bm}/failed"; continue; }
  echo "[W] ${cell} ${bm} start $(date -Is) on $(hostname)"
  ARGS=(--model_path "${MODEL_PATH}" --benchmarks "${bm}" --batch_size 1
        --gen_cache "${out}/gens" --output "${out}/${bm}.json")
  if [[ "${weights}" == "dense" ]]; then ARGS+=(--mode uncompressed)
  else ARGS+=(--mode traject_svd --weights_path "${weights}"); fi
  "$PYTHON_BIN" utils/eval_benchmarks.py "${ARGS[@]}" > "${out}/${bm}.log" 2>&1
  rc=$?
  # A json with acc=null would look "done" forever; drop it and mark failure.
  if [[ $rc -ne 0 ]] || ! "$PYTHON_BIN" -c "
import json,sys
d=json.load(open('${out}/${bm}.json'))
r=list(d.get('results',{}).values())
sys.exit(0 if r and r[0].get('acc') is not None else 1)" 2>/dev/null; then
    echo "[W] ${cell} ${bm} FAILED rc=${rc}; see ${out}/${bm}.log" >&2
    rm -f "${out}/${bm}.json"
    touch "${QUEUE_DIR}/claims/${cell}__${bm}/failed"
    continue
  fi
  n_done=$((n_done + 1))
  echo "[W] ${cell} ${bm} done $(date -Is)  ($(grep -o '"acc": [0-9.]*' "${out}/${bm}.json" | head -1))"
done
echo "[W] queue drained after ${n_done} task(s) $(date -Is)"
# Both breaks above land here: a genuinely empty queue, and one whose remaining
# lines are all claimed by peers that are still alive.
hold_or_exit "queue drained after ${n_done} task(s)"

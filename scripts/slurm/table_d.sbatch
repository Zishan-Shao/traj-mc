#!/usr/bin/env bash
# One (checkpoint arm x benchmark) cell of the LLaDA-Instruct low-rank table's
# HumanEval / MBPP / IFEval / BBH columns.
#
# Driven by scripts/run_table_d.sh, which exports CELL, WEIGHTS and BM.  Every
# sampler setting comes from utils/eval_benchmarks.py's BENCH_CONFIG, which
# follows the LLaDA repo's own OpenCompass configs per benchmark -- nothing
# about the protocol is decided here.
#SBATCH --job-name=tsvd-D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --output=results/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
SELF="${BASH_SOURCE[0]}"
PYTHON_BIN="${PYTHON_BIN:-/home/tl356/miniconda3/envs/AKS/bin/python}"
MODEL_PATH="${MODEL_PATH:-/zpool-00/home/tl356/LLaDA-8B-Instruct}"
PARTITION="${PARTITION//+/,}"
BBH_PER_SUBTASK="${BBH_PER_SUBTASK:-40}"
OUT="results/eval_stageB/${CELL}"
mkdir -p "$OUT"
if [ -f "$OUT/${BM}.json" ]; then echo "[D] ${CELL}/${BM} already done"; exit 0; fi

# One card in this partition raises "uncorrectable ECC error" as soon as a
# tensor touches it.  It fails in seconds and frees itself, so the scheduler
# hands it straight to the next job -- requeueing just lands back on it.
# Instead: hold this allocation so the faulty card leaves the pool, and
# relaunch the real work as a fresh job that will land on a healthy one.
if ! "$PYTHON_BIN" -c "import torch; x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite((x@x).sum())" 2>/dev/null; then
  echo "[D] GPU self-test failed on $(hostname) — quarantining this card and resubmitting ${CELL}/${BM}"
  sbatch --partition="${PARTITION}" --job-name="tD-${CELL}-${BM}" \
    --export=ALL,CELL="${CELL}",WEIGHTS="${WEIGHTS}",BM="${BM}" "${SELF}" || true
  sleep "${QUARANTINE_SECONDS:-86400}"
  exit 0
fi

echo "[D] ${CELL} ${BM} on $(hostname) $(date -Is)"
# batch_size=1 matches the official configs ('batch_size': 1) and the Stage B
# probe, where an 8B forward over a ~1700-token sequence already saturates the
# card and batching only paid for padding.
ARGS=(--model_path "${MODEL_PATH}" --benchmarks "${BM}" --batch_size 1
      --bbh_per_subtask "${BBH_PER_SUBTASK}"
      --gen_cache "${OUT}/gens" --output "$OUT/${BM}.json")
if [ "${WEIGHTS}" = "dense" ]; then ARGS+=(--mode uncompressed)
else ARGS+=(--mode traject_svd --weights_path "${WEIGHTS}"); fi
"$PYTHON_BIN" utils/eval_benchmarks.py "${ARGS[@]}"

# The harness writes its json even when a benchmark raised, leaving acc=null.
# Such a file satisfies the "already done" check at the top and would lock the
# cell out of ever being rerun, so drop it and fail loudly instead.  The
# generation cache is kept: a rerun then only pays for what is missing.
if ! "$PYTHON_BIN" -c "
import json,sys
d=json.load(open('$OUT/${BM}.json'))
r=list(d.get('results',{}).values())
sys.exit(0 if r and r[0].get('acc') is not None else 1)" 2>/dev/null; then
  echo "[D] ${CELL} ${BM} result invalid (acc=null); removing json so it can be rerun" >&2
  rm -f "$OUT/${BM}.json"
  exit 1
fi
echo "[D] ${CELL} ${BM} done $(date -Is)"

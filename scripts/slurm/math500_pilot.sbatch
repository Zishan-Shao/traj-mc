#!/usr/bin/env bash
# One cell of the MATH-500 protocol pilot: one arm at one sampler setting, on a
# fixed 100-problem prefix of MATH-500 (ids in results/pilot_math500/
# items_manifest.json).  Decides whether the finished MATH-500 column, which
# ran gen 512 / block 512, can stay or has to be rerun at the official
# 256 / 256 -- it is not itself a paper column.
#
# Driven by scripts/run_math500_pilot.sh, which exports CELL, WEIGHTS and CFG.
#SBATCH --job-name=tP-math500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/tl356/miniconda3/envs/AKS/bin/python}"
MODEL_PATH="${MODEL_PATH:-/zpool-00/home/tl356/LLaDA-8B-Instruct}"
OUT="results/pilot_math500"
mkdir -p "$OUT"
DEST="$OUT/${CELL}_${CFG}.json"
if [ -f "$DEST" ]; then echo "[P] ${CELL}/${CFG} already done"; exit 0; fi

# Same faulty-card guard the eval workers use.
if ! "$PYTHON_BIN" -c "import torch; x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite((x@x).sum())" 2>/dev/null; then
  echo "[P] GPU self-test failed on $(hostname); holding this card out of the pool"
  sleep "${QUARANTINE_SECONDS:-3600}"; exit 1
fi

# Slurm hands out one GPU per job, but it does not stop a process that ignores
# its own allocation from landing on ours: this job first ran three-way-shared
# with two foreign processes and went 244 s/question against 89 s/question
# alone.  Detect that at startup rather than discovering it in the timings, and
# hand the card back so the next attempt gets a clean one.
GPU_IDX="${CUDA_VISIBLE_DEVICES%%,*}"
FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
          -i "${GPU_IDX}" 2>/dev/null | wc -l)
if [ "${FOREIGN:-0}" -gt 0 ]; then
  echo "[P] GPU ${GPU_IDX} on $(hostname) already has ${FOREIGN} process(es) on it; resubmitting ${CELL}/${CFG}"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${GPU_IDX}" || true
  sbatch --partition="${SLURM_JOB_PARTITION}" --job-name="tP-${CELL}-${CFG}" \
    --export=ALL,CELL="${CELL}",WEIGHTS="${WEIGHTS}",CFG="${CFG}",GEN="${GEN}",BLK="${BLK}",STP="${STP}",LEI="${LEI}",CEEI="${CEEI}",LIMIT="${LIMIT:-300}" \
    "${BASH_SOURCE[0]}" || true
  sleep "${CONTENDED_HOLD_SECONDS:-600}"
  exit 0
fi

echo "[P] ${CELL} ${CFG} (gen=${GEN} block=${BLK} steps=${STP} L=${LEI} C=${CEEI}) on $(hostname) GPU ${GPU_IDX} $(date -Is)"
# No --gen_cache on purpose: the cache is keyed by question text alone, so the
# two sampler settings would collide on the same 100 items and the second one
# would silently replay the first one's answers.
ARGS=(--model_path "${MODEL_PATH}" --benchmarks math500 --batch_size 1
      --limit "${LIMIT:-300}" --gen_length "${GEN}" --block_length "${BLK}" --steps "${STP}"
      --logits_eos_inf "${LEI}" --confidence_eos_eot_inf "${CEEI}"
      --gen_diag --output "$DEST")
if [ "${WEIGHTS}" = "dense" ]; then ARGS+=(--mode uncompressed)
else ARGS+=(--mode traject_svd --weights_path "${WEIGHTS}"); fi
"$PYTHON_BIN" utils/eval_benchmarks.py "${ARGS[@]}"

if ! "$PYTHON_BIN" -c "
import json,sys
d=json.load(open('$DEST'))
r=list(d.get('results',{}).values())
sys.exit(0 if r and r[0].get('acc') is not None else 1)" 2>/dev/null; then
  echo "[P] ${CELL} ${CFG} invalid (acc=null); removing json" >&2
  rm -f "$DEST"; exit 1
fi
echo "[P] ${CELL} ${CFG} done $(date -Is)"

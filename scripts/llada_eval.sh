#!/usr/bin/env bash
# LLaDA-8B-Instruct benchmarks for Table 1.
#
# Protocols follow LLaDA's own OpenCompass configuration per benchmark
# (utils/eval_benchmarks.py): generative evaluation with the official
# low-confidence block sampler; GSM8K at gen 256 / block 8.
#
#   bash scripts/llada_eval.sh
#   CELLS="r0.8/traj" BENCHMARKS="gsm8k math500" bash scripts/llada_eval.sh
#
# One process per (cell, benchmark); results land in
# results/llada_instruct/eval/<cell>/<benchmark>.json and finished cells are
# skipped.  Cells run sequentially on the visible GPU; to parallelise, launch
# several copies with disjoint CELLS / BENCHMARKS (e.g. one per GPU).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${RESULTS}/llada_instruct"
CELLS="${CELLS:-dense r0.8/weight_svd r0.8/clean r0.8/traj r0.6/weight_svd r0.6/clean r0.6/traj}"
BENCHMARKS="${BENCHMARKS:-gsm8k math500 svamp arc_c mmlu hellaswag piqa}"

for cell in ${CELLS}; do
  dest="${OUT}/eval/${cell//\//_}"
  mkdir -p "${dest}"
  if [[ "${cell}" == dense ]]; then
    MODE=(--mode uncompressed)
  else
    weights="${OUT}/${cell%%/*}/weights/${cell##*/}"
    [[ -f "${weights}/compression_summary.json" ]] \
      || { echo "missing ${weights} (run scripts/llada_compress.sh)" >&2; exit 1; }
    MODE=(--mode traject_svd --weights_path "${weights}")
  fi
  for bm in ${BENCHMARKS}; do
    if [[ -f "${dest}/${bm}.json" ]]; then echo "[skip] ${cell} ${bm}"; continue; fi
    echo "[eval] ${cell} ${bm}"
    "${PYTHON_BIN}" utils/eval_benchmarks.py "${MODE[@]}" \
      --model_path "${LLADA_PATH}" --benchmarks "${bm}" --batch_size 1 \
      --gen_cache "${dest}/gens" --output "${dest}/${bm}.json"
  done
done
echo "[llada-eval] done -> ${OUT}/eval"

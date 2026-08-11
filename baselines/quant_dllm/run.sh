#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${QUANT_DLLM_MODEL_DIR:-${SCRIPT_DIR}/../model}"
DEVICE="${DEVICE:-cuda:0}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi

case "${1:-}" in
  llada|llada-base)
    RUNNER="${SCRIPT_DIR}/run_arb_llada.py"
    MODEL_PATH="${MODEL_ROOT}/LLaDA-8B-Base"
    EXTRA_ARGS=(--disable_mask --no_mask_order 2 --slim --abmp_ratio 0.05)
    ;;
  llada-instruct)
    RUNNER="${SCRIPT_DIR}/run_arb_llada.py"
    MODEL_PATH="${MODEL_ROOT}/LLaDA-8B-Instruct"
    EXTRA_ARGS=(--disable_mask --no_mask_order 2 --slim --abmp_ratio 0.05)
    ;;
  llada-1.5)
    RUNNER="${SCRIPT_DIR}/run_arb_llada.py"
    MODEL_PATH="${MODEL_ROOT}/LLaDA-1.5"
    EXTRA_ARGS=(--disable_mask --no_mask_order 2 --slim --abmp_ratio 0.05)
    ;;
  dream|dream-base)
    RUNNER="${SCRIPT_DIR}/run_arb_dream.py"
    MODEL_PATH="${MODEL_ROOT}/Dream-v0-Base-7B"
    EXTRA_ARGS=(--seed_x 100 --disable_mask --no_mask_order 2 --slim --abmp_ratio 0.10)
    ;;
  dream-instruct)
    RUNNER="${SCRIPT_DIR}/run_arb_dream.py"
    MODEL_PATH="${MODEL_ROOT}/Dream-v0-Instruct-7B"
    EXTRA_ARGS=(--seed_x 100 --disable_mask --no_mask_order 2 --slim --abmp_ratio 0.10)
    ;;
  *)
    echo "Usage: $0 MODEL [extra arguments]"
    echo "Models: llada-base, llada-instruct, llada-1.5,"
    echo "        dream-base, dream-instruct"
    echo "Example: DEVICE=cuda:1 $0 dream-instruct --nsamples 1"
    exit 2
    ;;
esac

shift
"${PYTHON}" "${RUNNER}" "${MODEL_PATH}" c4 arb-rc \
  --blocksize 128 --salient_metric hessian --device "${DEVICE}" \
  --save --num_p 1 --order2_group --mcs_prefix_ratio 0.25 \
  "${EXTRA_ARGS[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 BACKEND TASK ARM [WEIGHTS_DIR] [extra eval.run args]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$1"
TASK="$2"
ARM="$3"
shift 3

ARGS=(--backend "${BACKEND}" --task "${TASK}" --arm "${ARM}")
if [[ "${ARM}" != "ref" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "WEIGHTS_DIR is required for base/ours evaluation" >&2
    exit 2
  fi
  ARGS+=(--weights "$1")
  shift
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  ARGS+=(--model_path "${MODEL_PATH}")
fi
ARGS+=(--num_processes "${NUM_PROCESSES:-1}")

cd "${ROOT_DIR}"
python -m eval.run "${ARGS[@]}" "$@"

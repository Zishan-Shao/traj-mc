#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LLADA_REPO="${LLADA_REPO:-${ROOT_DIR}/LLaDA}"
DREAM_REPO="${DREAM_REPO:-${ROOT_DIR}/Dream}"
LLADA_REVISION="${LLADA_REVISION:-b7e6c3565f194352da44b38dd12b6970597309f3}"
DREAM_REVISION="${DREAM_REVISION:-31f94a60d187e3fd481fee3bbc2c732eb94a879c}"

clone_if_missing() {
  local url="$1"
  local destination="$2"
  local revision="$3"
  if [[ -d "${destination}" ]]; then
    echo "Using existing checkout: ${destination}"
    return
  fi
  git clone --filter=blob:none --no-checkout "${url}" "${destination}"
  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
}

apply_if_needed() {
  local repository="$1"
  local target_file="$2"
  local patch_file="$3"
  if grep -q "model_source" "${repository}/${target_file}"; then
    echo "Loader adaptation already present: ${target_file}"
    return
  fi
  git -C "${repository}" apply --check "${patch_file}"
  git -C "${repository}" apply "${patch_file}"
  echo "Applied loader adaptation: ${target_file}"
}

clone_if_missing \
  "https://github.com/ML-GSAI/LLaDA.git" \
  "${LLADA_REPO}" \
  "${LLADA_REVISION}"
clone_if_missing \
  "https://github.com/HKUNLP/Dream.git" \
  "${DREAM_REPO}" \
  "${DREAM_REVISION}"

apply_if_needed \
  "${LLADA_REPO}" \
  "eval_llada.py" \
  "${ROOT_DIR}/patches/llada-eval-local-checkpoint.patch"
apply_if_needed \
  "${DREAM_REPO}" \
  "eval/eval.py" \
  "${ROOT_DIR}/patches/dream-eval-local-checkpoint.patch"

echo "Evaluation repositories are ready."

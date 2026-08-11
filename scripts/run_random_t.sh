#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-llada}"
CORPUS="${CORPUS:-c4}"
SEED="${SEED:-42}"
CALIB_DIR="${OUT_DIR:-${ROOT_DIR}/results/calib/${BACKEND}}"

"${ROOT_DIR}/scripts/calibrate_pair.sh" "${BACKEND}"

GIT_HASH="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo nogit)"
BASE_CALIB="$(find "${CALIB_DIR}" -maxdepth 1 -type f \
  -name "${BACKEND}_base_${CORPUS}_n*_s${SEED}_full_${GIT_HASH}_calib.pt" \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
OURS_CALIB="$(find "${CALIB_DIR}" -maxdepth 1 -type f \
  -name "${BACKEND}_ours_${CORPUS}_n*_s${SEED}_full_${GIT_HASH}_calib.pt" \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "${BASE_CALIB}" || -z "${OURS_CALIB}" ]]; then
  echo "Could not identify the paired calibration tensors in ${CALIB_DIR}" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/compress_arm.sh" "${BACKEND}" base "${BASE_CALIB}"
"${ROOT_DIR}/scripts/compress_arm.sh" "${BACKEND}" ours "${OURS_CALIB}"

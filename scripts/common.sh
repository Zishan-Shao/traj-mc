# shellcheck shell=bash
# Shared settings and helpers for the reproduction scripts.  Sourced, not run.
#
# Every setting can be overridden from the environment, e.g.
#   PYTHON_BIN=/path/to/python LLADA_PATH=/models/LLaDA-8B-Instruct bash scripts/llada_compress.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LLADA_PATH="${LLADA_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-v0-Instruct-7B}"
RESULTS="${RESULTS:-${ROOT_DIR}/results}"

# Calibration protocol shared by every arm (paper, Appendix B): 256 C4 windows
# of 2048 tokens, window seed 42.  Clean, Traj-MC and fixed-t arms differ only
# in how those identical windows are masked.
NSAMPLES="${NSAMPLES:-256}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-42}"
CALIB_COMMON=(--corpus c4 --c4_streaming --seqlen "${SEQLEN}")

# calib_path ARGV... -> the artifact path trajmc.calibration writes for ARGV.
# The file name embeds the current git hash, so an artifact built at another
# commit is found by globbing over that slot.
calib_path() {
  local expected pattern matches=()
  expected="$("${PYTHON_BIN}" -m trajmc.calibration "$@" --print_artifact_path)"
  pattern="$(printf '%s' "${expected}" | sed -E 's/_[0-9a-zA-Z]{4,40}_calib\.pt$/_*_calib.pt/')"
  shopt -s nullglob
  matches=(${pattern})   # deliberately unquoted: this is a pattern
  shopt -u nullglob
  case ${#matches[@]} in
    0) printf '%s' "${expected}" ;;
    1) printf '%s' "${matches[0]}" ;;
    *) echo "ambiguous calibration artifact for ${expected}:" >&2
       printf '  %s\n' "${matches[@]}" >&2
       return 1 ;;
  esac
}

# build_calib ARGV... -> builds the artifact unless it exists; prints its path.
build_calib() {
  local path
  path="$(calib_path "$@")"
  if [[ ! -f "${path}" ]]; then
    "${PYTHON_BIN}" -m trajmc.calibration "$@" >&2
    path="$(calib_path "$@")"
    [[ -f "${path}" ]] || { echo "calibration did not produce ${path}" >&2; return 1; }
  fi
  printf '%s' "${path}"
}

# manifest_for CALIB_PT -> the sidecar json written next to it.
manifest_for() {
  printf '%s' "${1/_calib.pt/_manifest.json}"
}

# compress ARGS... -> runs trajmc.compression unless --save_path is complete.
compress() {
  local save="" prev=""
  for a in "$@"; do [[ "${prev}" == "--save_path" ]] && save="${a}"; prev="${a}"; done
  if [[ -n "${save}" && -f "${save}/compression_summary.json" ]]; then
    echo "[skip] ${save} already compressed"
    return 0
  fi
  "${PYTHON_BIN}" -m trajmc.compression "$@"
}

# E_gen holds the dense model plus one bin of factors; 4 bins fit a 24 GB card.
recon_bins() {
  local mib
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [[ -n "${mib}" ]] && (( mib >= 40000 )); then echo 1; else echo 4; fi
}

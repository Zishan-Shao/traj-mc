# shellcheck shell=bash
# Artifact-path resolution shared by the calibration scripts.
#
# trajmc/calibration.py owns the file-name format.  Re-deriving it in shell is
# what let the scheme suffix (_p0p25, _steps256) and the split-seed suffix
# (ws42_ms7) drift out of sync, and the mismatch only surfaced *after* the
# rollout it named had already been paid for.  Nothing here parses names: the
# module is asked for the path it would write, using the same argv as the run.

PYTHON_BIN="${PYTHON_BIN:-python}"

# calib_artifact_path ARGV...
# ARGV must be exactly the argv the real invocation will use, so the answer
# cannot disagree with the file that run goes on to write.
calib_artifact_path() {
  "${PYTHON_BIN}" -m trajmc.calibration "$@" --print_artifact_path
}

# require_calib_artifact PATH LABEL
require_calib_artifact() {
  local path="$1" label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing ${label} calibration artifact:" >&2
    echo "  ${path}" >&2
    echo "run the calibrate stage first, or check SEED/PREFIX_RATIO/ROLLOUT_STEPS" >&2
    return 1
  fi
}

# manifest_for CALIB_PT -- the sidecar json trajmc-calibrate writes next to it.
manifest_for() {
  printf '%s' "${1/_calib.pt/_manifest.json}"
}

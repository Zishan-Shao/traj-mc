#!/usr/bin/env bash
#SBATCH --job-name=trajsvd-stage-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=results/slurm/%x-%j.out

# Traj-SVD Stage A: five matched arms plus the two gates, one model, one ratio.
# Resumable -- rerun with STAGE=compress or STAGE=measure to pick up from the
# artifacts an earlier submission already wrote.
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${ROOT_DIR}/results/slurm"
cd "${ROOT_DIR}"

# Same guard the eval jobs carry: Slurm gives this job one GPU but cannot stop
# a process that ignores its own allocation from sitting on it too, and a
# shared card ran 2.7x slower in measurement.
# Hand this card back and queue a replacement -- but only if no other job of
# the same name is already waiting.  Without that check, cancelling a batch
# while some of its jobs sit in a guard block leaves each of them spawning a
# duplicate, and two jobs then write the same weights directory.
resubmit_self() {
  local live
  live="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -cxF "${SLURM_JOB_NAME}" || true)"
  if [ "${live:-0}" -gt 1 ]; then
    echo "[stage-a] ${SLURM_JOB_NAME} already has another instance queued; not resubmitting"
  else
    sbatch --partition="${SLURM_JOB_PARTITION}" --job-name="${SLURM_JOB_NAME}" \
      --time="${SLURM_TIMELIMIT:-01:30:00}" --export=ALL "${BASH_SOURCE[0]}" || true
  fi
  sleep "$1"
}

GPU_IDX="${CUDA_VISIBLE_DEVICES%%,*}"
# One card in this pool raises "uncorrectable ECC error" as soon as a tensor
# touches it.  It fails in seconds and frees itself, so the scheduler hands it
# straight to the next job; hold the allocation instead so the faulty card
# leaves the pool, and relaunch the real work as a fresh job.
if ! "${PYTHON_BIN:-python}" -c "import torch; x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite((x@x).sum())" 2>/dev/null; then
  echo "[stage-a] GPU self-test failed on $(hostname) GPU ${GPU_IDX} — quarantining it and resubmitting"
  resubmit_self "${QUARANTINE_SECONDS:-86400}"
  exit 0
fi

if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPU_IDX}" 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[stage-a] GPU ${GPU_IDX} on $(hostname) already busy with another process; resubmitting"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${GPU_IDX}" || true
  resubmit_self "${CONTENDED_HOLD_SECONDS:-600}"
  exit 0
fi

echo "[stage-a] node=$(hostname) GPU ${GPU_IDX} job=${SLURM_JOB_ID:-none} started $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

"${ROOT_DIR}/scripts/run_stage_a.sh" "${BACKEND:-llada_instruct}"
echo "[stage-a] finished $(date -Is)"

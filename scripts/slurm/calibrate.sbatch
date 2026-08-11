#!/usr/bin/env bash
#SBATCH --job-name=trajmc-calib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${ROOT_DIR}/results/slurm"
"${ROOT_DIR}/scripts/calibrate_pair.sh" "${BACKEND:-llada}"

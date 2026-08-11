#!/usr/bin/env bash
#SBATCH --job-name=trajmc-compress
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
: "${ARM:?submit with --export=ALL,ARM=base|ours,CALIB=/path/to/calib.pt}"
: "${CALIB:?submit with --export=ALL,ARM=base|ours,CALIB=/path/to/calib.pt}"
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${ROOT_DIR}/results/slurm"
"${ROOT_DIR}/scripts/compress_arm.sh" "${BACKEND:-llada}" "${ARM}" "${CALIB}"

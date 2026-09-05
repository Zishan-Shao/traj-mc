#!/usr/bin/env bash
# One (arm x task) cell of the Dream-Instruct table, under the OFFICIAL Dream
# Instruct protocol: the vendored eval_instruct/ toolkit (--model diffllm,
# --apply_chat_template, that fork's own task yamls).  Driven by
# scripts/run_dream_instruct_eval.sh, which exports TASK, FEWSHOT, GENLEN,
# SAMPLER, CELL and WEIGHTS.
#
# Dense and every compressed arm run this identical command; the ONLY thing
# that differs is the weights_path model_arg.
#SBATCH --job-name=dreamE
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --output=results/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/tl356/miniconda3/envs/llada/bin/python}"
HARNESS_DIR="${HARNESS_DIR:-baselines/dream_eval_instruct}"
MODEL_PATH="${MODEL_PATH:-/zpool-00/home/tl356/Dream-v0-Instruct-7B}"
# NGPU>1 runs lm-eval's own data parallelism: accelerate shards the request
# list across processes and the harness gathers them, so the protocol and the
# per-item results are unchanged -- only wall clock moves.  The official script
# uses `accelerate launch` the same way (it just does not cap num_processes).
NGPU="${NGPU:-1}"
# LIMIT caps each task at its first N documents.  For a group like
# mmlu_generative that is N per subject, so the 57-subject macro average the
# official metric reports keeps its structure; the prefix is deterministic, so
# every arm sees the same items and the comparison stays paired.
LIMIT="${LIMIT:-}"
# A capped run is a different measurement from a full one, so it gets its own
# directory rather than overwriting it.
OUT="results/eval_dream_instruct/${CELL}/${TASK}${LIMIT:+_lim${LIMIT}}"
# lm-eval writes to $OUT/<sanitised model name>/results_*.json, one level down.
if [ -n "$(find "$OUT" -name "results_*.json" 2>/dev/null | head -1)" ]; then
  echo "[E] ${CELL}/${TASK} already done"; exit 0
fi
mkdir -p "$OUT"

# With NGPU>1 the guards below check the first device only; a shared card
# anywhere in the set still shows up as one slow shard, which the progress bar
# makes visible.
GPU_IDX="${CUDA_VISIBLE_DEVICES%%,*}"
if ! "$PYTHON_BIN" -c "import torch; x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite((x@x).sum())" 2>/dev/null; then
  echo "[E] GPU self-test failed on $(hostname) GPU ${GPU_IDX}; quarantining"
  if [ "$(squeue -u "${USER}" -h -o '%j' | grep -cxF "${SLURM_JOB_NAME}")" -le 1 ]; then
    sbatch --partition="${SLURM_JOB_PARTITION}" --job-name="${SLURM_JOB_NAME}" \
      --time="${SLURM_TIMELIMIT:-24:00:00}" --export=ALL "${BASH_SOURCE[0]}" || true
  fi
  sleep "${QUARANTINE_SECONDS:-86400}"; exit 0
fi
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPU_IDX}" 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[E] GPU ${GPU_IDX} on $(hostname) already has another process; handing it back"
  if [ "$(squeue -u "${USER}" -h -o '%j' | grep -cxF "${SLURM_JOB_NAME}")" -le 1 ]; then
    sbatch --partition="${SLURM_JOB_PARTITION}" --job-name="${SLURM_JOB_NAME}" \
      --time="${SLURM_TIMELIMIT:-24:00:00}" --export=ALL "${BASH_SOURCE[0]}" || true
  fi
  sleep "${CONTENDED_HOLD_SECONDS:-600}"; exit 0
fi

# model_args, assembled exactly as eval_instruct/eval.sh writes them.  gpqa is
# the one row there that passes no sampler at all, so SAMPLER is empty for it
# and the harness defaults stand.
ARGS="pretrained=${MODEL_PATH},trust_remote_code=True"
if [ -n "${GENLEN}" ]; then
  ARGS="${ARGS},max_new_tokens=${GENLEN},diffusion_steps=${GENLEN}"
fi
ARGS="${ARGS},dtype=bfloat16"
# SAMPLER arrives '+'-separated because sbatch --export splits on commas.
[ -n "${SAMPLER}" ] && ARGS="${ARGS},${SAMPLER//+/,}"
[ "${WEIGHTS}" != "dense" ] && ARGS="${ARGS},weights_path=${WEIGHTS}"

echo "[E] ${CELL} ${TASK} on $(hostname) GPU ${GPU_IDX} $(date -Is)"
echo "[E] model_args: ${ARGS}"
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="${HARNESS_DIR}"
# GPQA is gated on the Hub; datasets only sees the credential through the env.
[ -f "${HOME}/.cache/huggingface/token" ] && export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
PORT=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))
CMD=("$PYTHON_BIN" -m accelerate.commands.launch --num_processes "${NGPU}"
     --main_process_port "$PORT" -m lm_eval
     --model diffllm --model_args "${ARGS}" --tasks "${TASK}"
     --device cuda --batch_size "${BATCHSZ:-1}" --num_fewshot "${FEWSHOT}"
     --output_path "$OUT" --log_samples --confirm_run_unsafe_code)
# HellaSwag and PIQA come from Dream's BASE mc script, which passes no chat
# template; every eval_instruct row does.  CHAT carries which one this task is.
[ "${CHAT:-1}" = "1" ] && CMD+=(--apply_chat_template)
[ -n "${LIMIT}" ] && CMD+=(--limit "${LIMIT}")
echo "[E] ${CMD[*]}"
"${CMD[@]}"
rc=$?
if [ $rc -ne 0 ] || [ -z "$(find "$OUT" -name "results_*.json" 2>/dev/null | head -1)" ]; then
  echo "[E] ${CELL} ${TASK} FAILED rc=$rc" >&2; exit 1
fi
echo "[E] ${CELL} ${TASK} done $(date -Is)"

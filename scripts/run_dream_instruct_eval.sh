#!/usr/bin/env bash
# Dream-Instruct table, under the OFFICIAL Dream Instruct protocol.
#
#   scripts/run_dream_instruct_eval.sh                      # every missing cell
#   TASKS=gsm8k_cot CELLS=dense scripts/run_dream_instruct_eval.sh
#   DRY=1 scripts/run_dream_instruct_eval.sh
#
# The protocol is the vendored baselines/dream_eval_instruct/ toolkit -- the
# repo's eval_instruct/ directory, which the root README's Evaluation section
# does not mention (it points only at eval/, the BASE protocol).  It is a
# separate lm-evaluation-harness fork: model class `diffllm`, its own task
# yamls (mmlu_generative, humaneval_instruct, mbpp_instruct), and
# --apply_chat_template on every task without exception.  Adding the flag to
# the Base harness would NOT be equivalent; the task configs differ too.
#
# Rows below are copied line for line from eval_instruct/eval.sh.  Shared
# sampler: temperature=0.1, top_p=0.9, alg=entropy, bfloat16, batch_size 1,
# max_new_tokens == diffusion_steps.  gpqa_main_n_shot is the one row that
# passes no sampler at all -- harness defaults stand there, so SAMPLER is empty.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-athena-genai,athena-small,athena,athena-mini}"
STAGE_ROOT="${STAGE_ROOT:-results/stage_a}"
MODEL="${MODEL:-dream_instruct}"
DRY="${DRY:-0}"
SAMPLER_COMMON='temperature=0.1,top_p=0.9,alg=entropy'

# task -> fewshot | genlen | extra model_args | batch_size | chat_template | timelimit
#
# Two protocols live in this table and the difference is Dream's, not ours:
#   * eval_instruct/eval.sh  -- generative, --apply_chat_template, batch 1.
#     Rows copied line for line from that script.
#   * eval/eval_dream_mc.sh  -- HellaSwag and PIQA exist ONLY there (the
#     Instruct toolkit ships no yaml for them), so they keep that script's
#     protocol: 0-shot, batch 32, add_bos_token=true, conditional likelihood,
#     and NO chat template.  Marked [base] below; the table note must say so.
declare -A TASK_SPEC=(
  [mmlu_generative]="4|128|${SAMPLER_COMMON}|1|1|24:00:00"
  [mmlu_pro]="4|128|${SAMPLER_COMMON}|1|1|24:00:00"
  [gsm8k_cot]="0|256|${SAMPLER_COMMON}|1|1|24:00:00"
  [minerva_math]="0|512|${SAMPLER_COMMON}|1|1|48:00:00"
  [gpqa_main_n_shot]="5|||1|1|12:00:00"
  [humaneval_instruct]="0|768|${SAMPLER_COMMON}|1|1|12:00:00"
  [mbpp_instruct]="0|1024|${SAMPLER_COMMON}|1|1|24:00:00"
  [ifeval]="0|1280|${SAMPLER_COMMON}|1|1|24:00:00"
  # [trajmc] MATH-500 on the official minerva_math row's sampler; the plan's
  # table reports MATH-500 and full MATH costs 10x (51 h/arm measured).
  [minerva_math500]="0|512|${SAMPLER_COMMON}|1|1|24:00:00"
  # [base] official Dream MCQ protocol, from eval/eval_dream_mc.sh
  [hellaswag]="0||add_bos_token=true|32|0|24:00:00"
  [piqa]="0||add_bos_token=true|32|0|12:00:00"
)

# MCS is not a row in this table (dropped, as it was for LLaDA), and its Dream
# factors were deleted to reclaim ~60G, so it is not reachable here at all.
declare -A CELL_WEIGHTS=(
  [dense]=dense
  [r08_weight_svd]="${STAGE_ROOT}/${MODEL}_r0.8/weights/weight_svd"
  [r08_clean]="${STAGE_ROOT}/${MODEL}_r0.8/weights/clean"
  [r08_traj]="${STAGE_ROOT}/${MODEL}_r0.8/weights/traj"
  [r06_weight_svd]="${STAGE_ROOT}/${MODEL}_r0.6/weights/weight_svd"
  [r06_clean]="${STAGE_ROOT}/${MODEL}_r0.6/weights/clean"
  [r06_traj]="${STAGE_ROOT}/${MODEL}_r0.6/weights/traj"
)

IFS=',' read -r -a TASK_LIST <<< "${TASKS:-mmlu_generative,gsm8k_cot,minerva_math500,gpqa_main_n_shot,hellaswag,piqa}"
# MCS is not a row in the plan's table (LLaDA dropped it), so the default
# cell set is the seven the table needs; r0?_mcs stays reachable by name.
IFS=',' read -r -a CELL_LIST <<< "${CELLS:-dense,r08_weight_svd,r08_clean,r08_traj,r06_weight_svd,r06_clean,r06_traj}"

QUEUED="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
n=0
for task in "${TASK_LIST[@]}"; do
  spec="${TASK_SPEC[${task}]:-}"
  [[ -n "${spec}" ]] || { echo "unknown task ${task}; known: ${!TASK_SPEC[*]}" >&2; exit 2; }
  IFS='|' read -r fewshot genlen sampler batchsz chat tlimit <<< "${spec}"
  # sbatch --export separates variables with commas, so a model_args string
  # containing them loses everything after the first one (this silently dropped
  # top_p=0.9 and alg=entropy from every generative run before it was caught).
  # Ship it comma-free and let the sbatch put the commas back.
  for cell in "${CELL_LIST[@]}"; do
    weights="${CELL_WEIGHTS[${cell}]:-}"
    [[ -n "${weights}" ]] || { echo "unknown cell ${cell}" >&2; exit 2; }
    if [[ "${weights}" != dense && ! -f "${weights}/compression_summary.json" ]]; then
      echo "[skip] ${cell} ${task}: weights not compressed yet"; continue
    fi
    out="results/eval_dream_instruct/${cell}/${task}${LIMIT:+_lim${LIMIT}}"
    # lm-eval nests its results under a sanitised model-name directory.
    if [ -n "$(find "${out}" -name "results_*.json" 2>/dev/null | head -1)" ]; then
      echo "[done]   ${cell} ${task}"; continue
    fi
    name="dE-${cell}-${task}${LIMIT:+-l${LIMIT}}"
    if grep -qxF "${name}" <<< "${QUEUED}"; then echo "[queued] ${cell} ${task}"; continue; fi
    echo "[submit] ${cell} ${task}  fs=${fewshot} len=${genlen:-n/a} bs=${batchsz} chat=${chat} gpus=${NGPU:-1} limit=${LIMIT:-full} ${sampler:-<harness defaults>}"
    n=$((n + 1))
    [[ "${DRY}" == 1 ]] && continue
    sbatch --partition="${PARTITION}" --nodes=1 --time="${tlimit}" --job-name="${name}" \
      --gres=gpu:"${NGPU:-1}" \
      --export=ALL,CELL="${cell}",WEIGHTS="${weights}",TASK="${task}",FEWSHOT="${fewshot}",GENLEN="${genlen}",SAMPLER="${sampler//,/+}",BATCHSZ="${batchsz}",CHAT="${chat}",NGPU="${NGPU:-1}",LIMIT="${LIMIT:-}" \
      scripts/slurm/dream_instruct_eval.sbatch
  done
done
echo "[run_dream_instruct_eval] submitted ${n}"

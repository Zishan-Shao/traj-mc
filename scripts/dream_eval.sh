#!/usr/bin/env bash
# Dream-7B-Instruct benchmarks for Table 1.
#
# Uses Dream's official Instruct evaluation toolkit (vendored in
# utils/dream_eval_instruct/, an lm-evaluation-harness fork with the `diffllm`
# model class) with rows copied from its eval.sh: chat template, batch 1,
# temperature 0.1, top_p 0.9, alg=entropy, max_new_tokens == diffusion_steps.
# HellaSwag, PIQA and ARC-C have no Instruct row upstream, so they use Dream's
# Base multiple-choice protocol (0-shot log-likelihood, no chat template).
# SVAMP has no Dream row at all and is scored with the gsm8k_cot row.
# Compressed arms differ from dense only by the `weights_path` model argument.
#
#   pip install -e "utils/dream_eval_instruct[ifeval,math]"
#   bash scripts/dream_eval.sh
#   CELLS="nokv_traj" TASKS="gsm8k_cot" NGPU=2 bash scripts/dream_eval.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${RESULTS}/dream_instruct"
W="${OUT}/r0.8/weights"
CELLS="${CELLS:-dense nokv_weight_svd nokv_clean nokv_traj}"
TASKS="${TASKS:-gsm8k_cot minerva_math500 svamp_cot arc_challenge mmlu_generative hellaswag piqa}"
NGPU="${NGPU:-1}"
SAMPLER='temperature=0.1,top_p=0.9,alg=entropy'

# task -> fewshot | gen length | sampler | batch size | chat template | limit
# GSM8K is scored on the first 512 test items and MMLU-generative on the first
# 40 items of each of its 57 subjects, as in the paper.
declare -A SPEC=(
  [gsm8k_cot]="0|256|${SAMPLER}|1|1|512"
  [minerva_math500]="0|512|${SAMPLER}|1|1|"
  [svamp_cot]="0|256|${SAMPLER}|1|1|"
  [mmlu_generative]="4|128|${SAMPLER}|1|1|40"
  [arc_challenge]="0||add_bos_token=true|32|0|"
  [hellaswag]="0||add_bos_token=true|32|0|"
  [piqa]="0||add_bos_token=true|32|0|"
)

export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="${ROOT_DIR}/utils/dream_eval_instruct:${PYTHONPATH}"
for task in ${TASKS}; do
  spec="${SPEC[${task}]:-}"
  [[ -n "${spec}" ]] || { echo "unknown task ${task}; known: ${!SPEC[*]}" >&2; exit 2; }
  IFS='|' read -r fewshot genlen sampler batch chat limit <<< "${spec}"
  for cell in ${CELLS}; do
    if [[ "${cell}" != dense ]]; then
      [[ -f "${W}/${cell}/compression_summary.json" ]] \
        || { echo "missing ${W}/${cell} (run scripts/dream_compress.sh)" >&2; exit 1; }
    fi
    dest="${OUT}/eval/${cell}/${task}"
    if [[ -n "$(find "${dest}" -name 'results_*.json' 2>/dev/null | head -1)" ]]; then
      echo "[skip] ${cell} ${task}"; continue
    fi
    args="pretrained=${DREAM_PATH},trust_remote_code=True"
    [[ -n "${genlen}" ]] && args+=",max_new_tokens=${genlen},diffusion_steps=${genlen}"
    args+=",dtype=bfloat16"
    [[ -n "${sampler}" ]] && args+=",${sampler}"
    [[ "${cell}" != dense ]] && args+=",weights_path=${W}/${cell}"
    CMD=("${PYTHON_BIN}" -m accelerate.commands.launch --num_processes "${NGPU}"
         -m lm_eval --model diffllm --model_args "${args}" --tasks "${task}"
         --device cuda --batch_size "${batch}" --num_fewshot "${fewshot}"
         --output_path "${dest}" --log_samples --confirm_run_unsafe_code)
    [[ "${chat}" == 1 ]] && CMD+=(--apply_chat_template)
    [[ -n "${limit}" ]] && CMD+=(--limit "${limit}")
    echo "[eval] ${cell} ${task}"
    "${CMD[@]}"
  done
done
echo "[dream-eval] done -> ${OUT}/eval"

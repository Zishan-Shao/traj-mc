#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/.."
MODEL_ROOT="${QUANT_DLLM_MODEL_DIR:-${WORKSPACE_DIR}/model}"
OUTPUT_ROOT="${QUANT_DLLM_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
EVAL_ROOT="${QUANT_DLLM_EVAL_DIR:-${SCRIPT_DIR}/eval_results/paper_table1}"
LLADA_REPO="${LLADA_REPO:-${SCRIPT_DIR}/LLaDA}"
DREAM_REPO="${DREAM_REPO:-${SCRIPT_DIR}/Dream}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi

model_key="${1:?Missing model argument}"
precision="${2:?Missing precision argument (fp or quant)}"
task_group="${3:?Missing task group argument}"
gpu="${4:?Missing GPU index}"

case "${model_key}" in
  llada-base)
    family=llada
    source_model="${MODEL_ROOT}/LLaDA-8B-Base"
    quant_model="${OUTPUT_ROOT}/LLaDA-8B-Base_c4_arb-rc_128_hessian_nump_1_order2group_True_gptaq_False_disable_mask_True_no_mask_order_2_slim_True_abmp_ratio_0.05_mcs_prefix_0.25_seed_0_maskx+salience+slim.pt"
    ;;
  llada-instruct)
    family=llada
    source_model="${MODEL_ROOT}/LLaDA-8B-Instruct"
    quant_model="${OUTPUT_ROOT}/LLaDA-8B-Instruct_c4_arb-rc_128_hessian_nump_1_order2group_True_gptaq_False_disable_mask_True_no_mask_order_2_slim_True_abmp_ratio_0.05_mcs_prefix_0.25_seed_0_maskx+salience+slim.pt"
    ;;
  llada-1.5)
    family=llada
    source_model="${MODEL_ROOT}/LLaDA-1.5"
    quant_model="${OUTPUT_ROOT}/LLaDA-1.5_c4_arb-rc_128_hessian_nump_1_order2group_True_gptaq_False_disable_mask_True_no_mask_order_2_slim_True_abmp_ratio_0.05_mcs_prefix_0.25_seed_0_maskx+salience+slim.pt"
    ;;
  dream-base)
    family=dream
    source_model="${MODEL_ROOT}/Dream-v0-Base-7B"
    quant_model="${OUTPUT_ROOT}/Dream-v0-Base-7B_c4_arb-rc_128_hessian_nump_1_order2group_True_gptaq_False_disable_mask_True_no_mask_order_2_slim_True_abmp_ratio_0.1_mcs_prefix_0.25_seed_100_maskx+salience+slim.pt"
    ;;
  dream-instruct)
    family=dream
    source_model="${MODEL_ROOT}/Dream-v0-Instruct-7B"
    quant_model="${OUTPUT_ROOT}/Dream-v0-Instruct-7B_c4_arb-rc_128_hessian_nump_1_order2group_True_gptaq_False_disable_mask_True_no_mask_order_2_slim_True_abmp_ratio_0.1_mcs_prefix_0.25_seed_100_maskx+salience+slim.pt"
    ;;
  *)
    echo "Unknown model: ${model_key}" >&2
    exit 2
    ;;
esac

quant_model="${QUANT_MODEL_PATH:-${quant_model}}"

case "${precision}" in
  fp)
    eval_model="${source_model}"
    source_arg=""
    ;;
  quant)
    eval_model="${quant_model}"
    source_arg=",model_source=${source_model}"
    if [[ ! -d "${eval_model}" ]]; then
      echo "Quantized checkpoint is not ready: ${eval_model}" >&2
      exit 3
    fi
    ;;
  *)
    echo "Precision must be fp or quant" >&2
    exit 2
    ;;
esac

case "${task_group}" in
  five_shot)
    tasks="mmlu,winogrande"
    num_fewshot=5
    ;;
  zero_shot)
    tasks="piqa,arc_challenge,arc_easy,hellaswag"
    num_fewshot=0
    ;;
  bbh)
    tasks="bbh"
    num_fewshot=0
    ;;
  *)
    echo "Task group must be five_shot, zero_shot, or bbh" >&2
    exit 2
    ;;
esac

result_dir="${EVAL_ROOT}/${precision}/${model_key}/${task_group}"
mkdir -p "${result_dir}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_MODULES_CACHE="${WORKSPACE_DIR}/.hf_modules/table1_gpu${gpu}"

if [[ "${family}" == "llada" ]]; then
  if [[ ! -f "${LLADA_REPO}/eval_llada.py" ]]; then
    echo "LLaDA evaluation repository not found: ${LLADA_REPO}" >&2
    exit 4
  fi
  model_args="model_path=${eval_model}${source_arg},cfg=0.0,is_check_greedy=False"
  if [[ "${task_group}" == "bbh" ]]; then
    model_args+=",gen_length=1024,steps=1024,block_length=1024"
    batch_size=1
  else
    model_args+=",mc_num=128"
    batch_size=8
  fi
  exec "${PYTHON}" "${LLADA_REPO}/eval_llada.py" \
    --model llada_dist \
    --model_args "${model_args}" \
    --tasks "${tasks}" \
    --num_fewshot "${num_fewshot}" \
    --batch_size "${batch_size}" \
    --output_path "${result_dir}"
else
  if [[ ! -f "${DREAM_REPO}/eval/eval.py" ]]; then
    echo "Dream evaluation repository not found: ${DREAM_REPO}" >&2
    exit 4
  fi
  model_args="pretrained=${eval_model}${source_arg},add_bos_token=true"
  if [[ "${task_group}" == "bbh" ]]; then
    model_args+=",max_new_tokens=512,diffusion_steps=512,temperature=0,top_p=0.95"
    batch_size=1
  else
    model_args+=",mc_num=128"
    batch_size=32
  fi
  exec "${PYTHON}" "${DREAM_REPO}/eval/eval.py" \
    --model dream \
    --model_args "${model_args}" \
    --tasks "${tasks}" \
    --num_fewshot "${num_fewshot}" \
    --batch_size "${batch_size}" \
    --output_path "${result_dir}" \
    --confirm_run_unsafe_code
fi

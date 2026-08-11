# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'en', split='train[:50_000]')
    valdata = load_dataset('allenai/c4', 'en', split='validation[:5_000]')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def _mmlu_example(row, include_answer=True):
    choices = row["choices"]
    ans_idx = int(row["answer"])
    text = f"Question: {row['question'].strip()}\n"
    for i, choice in enumerate(choices):
        text += f"{LETTERS[i]}. {choice}\n"
    text += f"Answer: {LETTERS[ans_idx]}\n" if include_answer else "Answer:"
    return text

def _build_mmlu_prompt(row, fewshot_rows):
    subject = row["subject"].replace("_", " ")
    parts = [f"The following are multiple choice questions (with answers) about {subject}.\n"]
    parts.extend(_mmlu_example(ex, include_answer=True) for ex in fewshot_rows)
    parts.append(_mmlu_example(row, include_answer=False))
    return "\n".join(parts)

def _build_gsm8k_prompt(row, fewshot_rows):
    parts = []
    for ex in fewshot_rows:
        parts.append(
            f"Question: {ex['question'].strip()}\n"
            f"Let's think step by step\n"
            f"Answer: {ex['answer'].strip()}"
        )
    parts.append(f"Question: {row['question'].strip()}\nLet's think step by step\nAnswer:")
    return "\n\n".join(parts)

def _chat_format(tokenizer, prompt):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt

def get_mmlu_gsm8k_prompts(nsamples, seed, seqlen, tokenizer):
    mmlu_target = load_dataset("cais/mmlu", "all", split="validation")
    mmlu_dev = load_dataset("cais/mmlu", "all", split="dev")
    gsm8k_train = list(load_dataset("gsm8k", "main", split="train"))

    by_subject = {}
    for row in mmlu_dev:
        by_subject.setdefault(row["subject"], []).append(row)

    prompts = []
    for row in mmlu_target:
        fewshot = by_subject.get(row["subject"], [])[:5]
        prompts.append(_build_mmlu_prompt(row, fewshot))

    gsm8k_fewshot = gsm8k_train[:4]
    for row in gsm8k_train[4:]:
        prompts.append(_build_gsm8k_prompt(row, gsm8k_fewshot))

    random.seed(seed)
    random.shuffle(prompts)
    text = "\n\n".join(_chat_format(tokenizer, prompt) for prompt in prompts)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids

    if input_ids.shape[1] <= seqlen:
        raise ValueError(
            f"Task prompt calibration corpus is too short for seqlen={seqlen}: "
            f"{input_ids.shape[1]} tokens"
        )

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    testenc = TokenizerWrapper(input_ids[:, :min(input_ids.shape[1], 256 * seqlen)])
    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "mmlu_gsm8k" in name or "task_prompts" in name:
        return get_mmlu_gsm8k_prompts(nsamples, seed, seqlen, tokenizer)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)

def get_loaders_llada(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2_llada(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4_llada(nsamples, seed, seqlen, tokenizer)

def get_loaders_dream(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2_dream(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4_dream(nsamples, seed, seqlen, tokenizer)

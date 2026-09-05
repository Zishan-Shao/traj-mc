"""LLaDA's official evaluation prompts, for the Actual-State rollout control.

The Actual-State arm answers "what if we calibrated on the states the dense
model really visits?".  Those states depend on the exact prompt the deployed
harness sends, so everything here is copied from
``utils/eval_benchmarks.py`` rather than reconstructed: the fixed four-shot
block, the ``Let's think step by step`` line, the ``\\n\\n`` join, the chat
template, ``add_special_tokens=False``, and the 1900-token prompt cap.

The official sampler settings live here too, so a caller cannot pass a
generation length or block length that disagrees with the harness.
"""

from __future__ import annotations

import os

import torch

from .gsm8k_fewshot import FEWSHOT as GSM8K_FEWSHOT

#: utils/eval_benchmarks.py::MAX_PROMPT_TOKENS
MAX_PROMPT_TOKENS = 1900

#: utils/eval_benchmarks.py::BENCH_CONFIG, verbatim.
OFFICIAL_SAMPLER = {
    "gsm8k": dict(gen_length=256, block_length=8, steps=256,
                  logits_eos_inf=False, confidence_eos_eot_inf=False,
                  num_fewshot=4),
    "math": dict(gen_length=512, block_length=512, steps=512,
                 logits_eos_inf=False, confidence_eos_eot_inf=True,
                 num_fewshot=4),
}

GSM8K = ("gsm8k", "main")
MATH500 = ("HuggingFaceH4/MATH-500", None)

#: name -> (dataset, split, task key in OFFICIAL_SAMPLER)
SOURCES = {
    "gsm8k_train": (GSM8K, "train", "gsm8k"),
    "gsm8k_test": (GSM8K, "test", "gsm8k"),
    "math500": (MATH500, "test", "math"),
}


def official_sampler(source: str) -> dict:
    """Deployed generation settings for a prompt source."""
    if source not in SOURCES:
        raise ValueError(f"unknown prompt source {source!r}; "
                         f"choose one of: {', '.join(sorted(SOURCES))}")
    return dict(OFFICIAL_SAMPLER[SOURCES[source][2]])


def build_gsm8k_prompt(question: str) -> str:
    """utils/eval_benchmarks.py::build_gsm8k_prompt, for one question."""
    parts = [
        f"Question: {q}\nLet's think step by step\nAnswer: {a}"
        for q, a in GSM8K_FEWSHOT
    ]
    parts.append(
        f"Question: {question.strip()}\nLet's think step by step\nAnswer:"
    )
    return "\n\n".join(parts)


def build_math_prompt(problem: str, shots: list[tuple[str, str]]) -> str:
    """Four-shot MATH prompt in the harness's Problem/Solution format."""
    parts = [f"Problem:\n{p}\n\nSolution: {s}" for p, s in shots]
    parts.append(f"Problem:\n{problem.strip()}\n\nSolution:")
    return "\n\n".join(parts)


def to_chat_text(tokenizer, prompt_text: str) -> str:
    """utils/eval_benchmarks.py::_to_chat_text."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def load_split(name: str):
    """Load one prompt source's split, preferring the local cache."""
    if name not in SOURCES:
        raise ValueError(f"unknown prompt source {name!r}")
    from datasets import load_dataset

    (path, config), split, _ = SOURCES[name]
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    return load_dataset(path, config, split=split) if config else \
        load_dataset(path, split=split)


def build_task_prompts(
    tokenizer,
    source: str,
    count: int,
    prompt_length: int | None,
    seed: int,
    chat_template: bool = True,
    num_fewshot: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Return ``[count, prompt_length]`` prompt tokens in the harness's format.

    GSM8K uses the harness's fixed four-shot block, so ``seed`` selects only
    the target questions and every row shares one context by construction.
    MATH draws its four shots from the same split, disjoint from the targets.

    Calibration needs a rectangular tensor while the harness sends variable
    lengths, so rows are left-truncated to a common length -- the shortest
    natural prompt among the selected targets -- which trims the head of the
    first example and never touches the target question or the template. No
    pad or EOS token is ever inserted: EOS padding in the SFT data is what
    makes this checkpoint flood EOS in the first place.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    dataset = load_split(source)
    task = SOURCES[source][2]
    shots_wanted = OFFICIAL_SAMPLER[task]["num_fewshot"] if num_fewshot is None \
        else num_fewshot

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()

    if task == "gsm8k":
        if shots_wanted != len(GSM8K_FEWSHOT):
            raise ValueError(
                f"the harness fixes GSM8K at {len(GSM8K_FEWSHOT)} shots; "
                f"got num_fewshot={shots_wanted}"
            )
        targets = order[:count]
        shot_indices = None
        texts = [build_gsm8k_prompt(dataset[i]["question"]) for i in targets]
    else:
        shot_indices = order[:shots_wanted]
        targets = order[shots_wanted : shots_wanted + count]
        if len(targets) < count:
            raise ValueError(f"{source} cannot supply {count} disjoint targets")
        shots = [(dataset[i]["problem"], dataset[i]["solution"])
                 for i in shot_indices]
        texts = [build_math_prompt(dataset[i]["problem"], shots) for i in targets]

    if chat_template:
        texts = [to_chat_text(tokenizer, text) for text in texts]
    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"]
               for text in texts]
    capped = sum(1 for ids in encoded if len(ids) > MAX_PROMPT_TOKENS)
    encoded = [ids[-MAX_PROMPT_TOKENS:] if len(ids) > MAX_PROMPT_TOKENS else ids
               for ids in encoded]

    natural = [len(ids) for ids in encoded]
    if prompt_length is None:
        prompt_length = min(natural)
    rows = []
    for ids in encoded:
        if len(ids) < prompt_length:
            raise ValueError(
                f"a prompt is {len(ids)} tokens, shorter than the requested "
                f"prompt_length={prompt_length}; padding is not allowed"
            )
        rows.append(ids[len(ids) - prompt_length :])

    prompts = torch.tensor(rows, dtype=torch.long)
    if prompts.shape != (count, prompt_length):
        raise RuntimeError(f"packed prompts have shape {tuple(prompts.shape)}")
    return prompts, {
        "prompt_source": source,
        "prompt_split": SOURCES[source][1],
        "prompt_task": task,
        "prompt_length": prompt_length,
        "prompt_chat_template": chat_template,
        "prompt_num_fewshot": shots_wanted,
        "prompt_fewshot_fixed": task == "gsm8k",
        "prompt_fewshot_indices": shot_indices,
        "prompt_target_indices": targets,
        "prompt_target_seed": seed,
        "prompt_natural_length_min": min(natural),
        "prompt_natural_length_max": max(natural),
        "prompt_rows_capped_at_max": capped,
        "prompt_source_rows": len(dataset),
        "prompt_builder": "utils/eval_benchmarks.py (verbatim)",
        "official_sampler": dict(OFFICIAL_SAMPLER[task]),
    }


def shared_prompt_length(tokenizer, specs, chat_template: bool = True,
                         num_fewshot: int | None = None) -> int:
    """Smallest natural prompt length across several ``(source, count, seed)``.

    A calibration source and an evaluation source must be trimmed to the *same*
    length or the control arm and the states it is judged on carry different
    mask geometry.  Taking each source's own minimum does not achieve that:
    the minimum depends on which target questions were drawn.
    """
    lengths = []
    for source, count, seed in specs:
        _, meta = build_task_prompts(
            tokenizer, source, count, None, seed,
            chat_template=chat_template, num_fewshot=num_fewshot,
        )
        lengths.append(meta["prompt_natural_length_min"])
    return min(lengths)


def main():
    """Print the shared prompt length for a set of sources."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--spec", action="append", required=True,
                        help="SOURCE:SEED:COUNT, repeatable")
    parser.add_argument("--no_chat_template", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    specs = []
    for raw in args.spec:
        source, seed, count = raw.split(":")
        specs.append((source, int(count), int(seed)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                             trust_remote_code=True)
    print(shared_prompt_length(tokenizer, specs,
                               chat_template=not args.no_chat_template))


if __name__ == "__main__":
    main()

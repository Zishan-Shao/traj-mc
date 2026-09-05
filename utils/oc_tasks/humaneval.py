"""HumanEval, on the protocol the LLaDA-8B-Instruct numbers were produced with.

``examples/llada_instruct_gen_humaneval_length512_block512_logits.py`` in the
LLaDA repo points at OpenCompass's ``humaneval_gen_8e312c``: zero-shot, one
user turn ``"Complete the following python code:\\n{prompt}"``, the reply run
through ``humaneval_postprocess_v2``, and pass@1 measured by appending the
completion to the problem prompt before the unit tests -- see
``open-compass/human-eval``'s ``check_correctness``.
"""

from __future__ import annotations

import re

from . import data
from .sandbox import run_programs

PROMPT = "Complete the following python code:\n{prompt}"
TIMEOUT = 3.0


def build(limit: int | None = None) -> list[dict]:
    rows = data.load_humaneval()
    if limit:
        rows = rows[:limit]
    return [{"messages": [{"role": "user",
                           "content": PROMPT.format(prompt=row["prompt"])}],
             "meta": row}
            for row in rows]


def postprocess(text: str) -> str:
    """OpenCompass ``humaneval_postprocess_v2``."""
    blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    if len(blocks) >= 1:
        text = blocks[0]
    return text.lstrip()


def score(items: list[dict], generations: list[str]) -> dict:
    programs, completions = [], []
    for item, generation in zip(items, generations):
        row = item["meta"]
        completion = postprocess(generation)
        completions.append(completion)
        programs.append(
            row["prompt"] + completion + "\n"
            + row["test"] + "\n"
            + f"check({row['entry_point']})"
        )
    outcomes = run_programs(programs, timeout=TIMEOUT)
    records = [
        {"idx": i, "gold": item["meta"]["task_id"], "pred": completion,
         "correct": outcome == "pass", "result": outcome}
        for i, (item, completion, outcome)
        in enumerate(zip(items, completions, outcomes))
    ]
    n_pass = sum(record["correct"] for record in records)
    return {"acc": n_pass / len(records) if records else 0.0,
            "n": len(records), "per_item": records}

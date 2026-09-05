"""IFEval, on the protocol the LLaDA-8B-Instruct numbers were produced with.

``examples/llada_instruct_gen_ifeval_length512_block512_confidence.py`` points
at OpenCompass's ``IFEval_gen`` -> ``IFEval_gen_353ae7``: zero-shot, the raw
prompt as the only user turn, judged by ``IFEvaluator``.  The judge itself is
vendored under ``ifeval/``.

Four numbers come out; the headline one reported as ``acc`` is prompt-level
strict accuracy, the strictest and the one usually quoted as "IFEval".
"""

from __future__ import annotations

from . import data
from .ifeval import (InputExample, test_instruction_following_loose,
                     test_instruction_following_strict)


def build(limit: int | None = None) -> list[dict]:
    rows = data.load_ifeval()
    if limit:
        rows = rows[:limit]
    return [{"messages": [{"role": "user", "content": row["prompt"]}],
             "meta": row}
            for row in rows]


def score(items: list[dict], generations: list[str]) -> dict:
    prompt_strict = inst_strict_hits = inst_strict_total = 0
    prompt_loose = inst_loose_hits = inst_loose_total = 0
    records = []
    for i, (item, generation) in enumerate(zip(items, generations)):
        reference = item["meta"]
        kwargs = [{k: v for k, v in kwarg.items() if v is not None}
                  for kwarg in reference["kwargs"]]
        example = InputExample(key=reference["key"],
                               instruction_id_list=reference["instruction_id_list"],
                               prompt=reference["prompt"],
                               kwargs=kwargs)
        strict = test_instruction_following_strict(example, generation)
        loose = test_instruction_following_loose(example, generation)

        is_strict = all(strict.follow_instruction_list)
        is_loose = all(loose.follow_instruction_list)
        prompt_strict += is_strict
        prompt_loose += is_loose
        inst_strict_hits += sum(strict.follow_instruction_list)
        inst_loose_hits += sum(loose.follow_instruction_list)
        inst_strict_total += len(strict.instruction_id_list)
        inst_loose_total += len(loose.instruction_id_list)
        records.append({"idx": i, "gold": reference["key"],
                        "pred": None, "correct": bool(is_strict),
                        "loose_correct": bool(is_loose)})

    n = len(records) or 1
    return {"acc": prompt_strict / n, "n": len(records), "per_item": records,
            "prompt_level_strict": prompt_strict / n,
            "prompt_level_loose": prompt_loose / n,
            "inst_level_strict": inst_strict_hits / max(inst_strict_total, 1),
            "inst_level_loose": inst_loose_hits / max(inst_loose_total, 1)}

"""BBH, on the protocol the LLaDA numbers were produced with.

``examples/llada_base_gen_bbh_length256_block256.py`` points at OpenCompass's
``bbh_gen`` -> ``bbh_gen_ee62e9``: one user turn per question carrying the
official three-shot chain-of-thought hint for that subtask, then the two
answer extractors (``bbh_mcq_postprocess`` for the seventeen multiple-choice
subtasks, ``bbh_freeform_postprocess`` for the ten free-form ones).  The hints
in ``bbh_prompts/`` are the config's ``lib_prompt/*.txt``, unmodified.

The headline number is the ``bbh`` summary group: the unweighted mean over the
twenty-seven subtask accuracies, not a pooled item accuracy.  The LLaDA repo
publishes no Instruct-specific BBH config, so the Base one's sampler settings
are what the runner uses here.
"""

from __future__ import annotations

import os
import re

from . import data

PROMPT = ("Follow the given examples and answer the question.\n{hint}\n\n"
          "Q: {input}\nA: Let's think step by step.")
_PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bbh_prompts")


def hint(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, f"{name}.txt"), encoding="utf-8") as handle:
        return handle.read()


def build(limit: int | None = None, per_subtask: int | None = None) -> list[dict]:
    """Questions from all 27 subtasks, subtask-major.

    ``per_subtask`` caps each subtask at its first N questions.  Taking a
    prefix rather than a random sample keeps the item set identical across
    arms (so the comparison stays paired) and makes a later, larger cap a
    strict superset of what has already been run.
    """
    items = []
    for name in data.BBH_SETS:
        rows = data.load_bbh(name)
        if per_subtask:
            rows = rows[:per_subtask]
        hint_text = hint(name)
        for row in rows:
            items.append({
                "messages": [{"role": "user", "content": PROMPT.format(
                    hint=hint_text, input=row["input"])}],
                "meta": {"subtask": name, "target": row["target"],
                         "mcq": name in data.BBH_MULTIPLE_CHOICE_SETS},
            })
    if limit:
        items = items[:limit]
    return items


def mcq_postprocess(text: str) -> str:
    """OpenCompass ``bbh_mcq_postprocess`` (applied to both pred and gold)."""
    ans = text
    ans_line = ans.split("answer is ")
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r"\(([A-Z])\)*", ans)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])", ans)
    if match:
        return match.group(1)
    return ans


def freeform_postprocess(text: str) -> str:
    """OpenCompass ``bbh_freeform_postprocess`` (applied to the pred only)."""
    ans = text
    ans_line = ans.split("answer is ")
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split("\n")[0].strip()
    if ans.endswith("."):
        ans = ans[:-1].strip()
    match = re.search(r"\*\*(.*?)\*\*", ans)
    if match:
        return match.group(1)
    return ans


def score(items: list[dict], generations: list[str]) -> dict:
    records, by_subtask = [], {}
    for i, (item, generation) in enumerate(zip(items, generations)):
        meta = item["meta"]
        if meta["mcq"]:
            pred = mcq_postprocess(generation)
            gold = mcq_postprocess(meta["target"])
        else:
            pred = freeform_postprocess(generation)
            gold = meta["target"]
        correct = pred == gold
        records.append({"idx": i, "subtask": meta["subtask"], "gold": gold,
                        "pred": pred, "correct": correct})
        hits, total = by_subtask.get(meta["subtask"], (0, 0))
        by_subtask[meta["subtask"]] = (hits + int(correct), total + 1)

    subtask_acc = {name: hits / total for name, (hits, total) in by_subtask.items()}
    # The 'bbh' summary group averages subtask accuracies, so a subtask with
    # fewer questions still counts once.
    acc = sum(subtask_acc.values()) / len(subtask_acc) if subtask_acc else 0.0
    return {"acc": acc, "n": len(records), "per_item": records,
            "subtask_acc": {name: round(value, 6)
                            for name, value in sorted(subtask_acc.items())},
            "pooled_acc": (sum(r["correct"] for r in records) / len(records)
                           if records else 0.0)}

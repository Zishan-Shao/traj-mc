"""MBPP, on the protocol the LLaDA-8B-Instruct numbers were produced with.

``examples/llada_instruct_gen_mbpp_length256_block256_confidence.py`` points at
OpenCompass's ``mbpp_gen`` -> ``mbpp_gen_830460``: a three-shot chat prompt
whose demonstrations are fixed strings (not sampled from a train split), a
final assistant turn pre-filled with ``[BEGIN]\\n``, ``MBPPEvaluator``'s
``_process_answer`` to recover the code, and the problem's own asserts as the
test.  The three demonstrations and the answer-extraction patterns below are
copied verbatim from that config and evaluator.
"""

from __future__ import annotations

import re

from . import data
from .sandbox import run_programs

TIMEOUT = 10.0

#: ``mbpp_gen_830460``'s prompt template, turn for turn.
FEWSHOT = [
    ("You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n",
     "[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n "),
    ("You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n",
     "[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n "),
    ("You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n",
     "[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n "),
]
QUERY = ("You are an expert Python programmer, and here is your task: {text} "
         "Your code should pass these tests:\n\n {test_list}  \n")
#: The config ends on a BOT turn holding only this, i.e. the model continues an
#: assistant message that has already opened the code block.
ANSWER_PREFIX = "[BEGIN]\n"

#: ``MBPPEvaluator._process_answer``, in order -- the first pattern that
#: matches wins, so the order matters.
_PATTERNS = [
    r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
    r"BEGIN\s*'(.*)'\s*\[DONE\]",
    r"\[BEGIN\]\s*'(.*)'\s*DONE",
    r"BEGIN\s*'(.*)'\s*DONE",
    r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
    r"BEGIN\s*'(.*)\s*\[DONE\]",
    r"\[BEGIN\]\s*'(.*)\s*DONE",
    r"BEGIN\s*'(.*)\s*DONE",
    r"\[BEGIN\]\s*(.*)\s*\[DONE\]",
    r"BEGIN\s*(.*)\s*\[DONE\]",
    r"\[BEGIN\]\s*(.*)\s*DONE",
    r"BEGIN\s*(.*)\s*DONE",
    r"```python\s*(.*)\s*```",
    r"```\s*(.*)\s*```",
    r"```python\s*(.*)\s*$",
    r"```\s*(.*)\s*$",
    r"(.*)\s*```.*",
    r"\[BEGIN\]\s*'(.*)",
    r"\[BEGIN\](.*)",
    r"'(.*)'\s*\[DONE\]",
]


def build(limit: int | None = None) -> list[dict]:
    rows = data.load_mbpp()
    if limit:
        rows = rows[:limit]
    items = []
    for row in rows:
        messages = []
        for user, assistant in FEWSHOT:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": QUERY.format(
            text=row["text"], test_list="\n".join(row["test_list"]))})
        items.append({"messages": messages, "answer_prefix": ANSWER_PREFIX,
                      "meta": {"task_id": int(row["task_id"]),
                               "test_list": list(row["test_list"])}})
    return items


def postprocess(text: str) -> str:
    """OpenCompass ``MBPPEvaluator._process_answer``."""
    for pattern in _PATTERNS:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            break
    text = text.split("```")[0]
    text = re.split(r"'?\s*\[?DONE\]?", text)[0]
    text = text.replace("\\_", "_")
    text = text.strip()
    if text.startswith("'"):
        text = text[1:]
    if text.endswith("'"):
        text = text[:-1]
    return text


def score(items: list[dict], generations: list[str]) -> dict:
    programs, codes = [], []
    for item, generation in zip(items, generations):
        # The model continues an assistant turn that already reads "[BEGIN]\n",
        # so put that back before extraction -- the patterns key off it.
        code = postprocess(item.get("answer_prefix", "") + generation)
        codes.append(code)
        programs.append(code + "\n" + "\n".join(item["meta"]["test_list"]))
    outcomes = run_programs(programs, timeout=TIMEOUT)
    records = [
        {"idx": i, "gold": item["meta"]["task_id"], "pred": code,
         "correct": outcome == "pass", "result": outcome}
        for i, (item, code, outcome) in enumerate(zip(items, codes, outcomes))
    ]
    n_pass = sum(record["correct"] for record in records)
    return {"acc": n_pass / len(records) if records else 0.0,
            "n": len(records), "per_item": records}

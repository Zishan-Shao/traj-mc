"""Dataset access for the OpenCompass-protocol benchmarks.

``datasets==2.16.1`` (the version this environment pins for LLaDA) calls
``dataclasses.asdict`` on the object ``huggingface_hub`` returns from
``dataset_info``; on the installed hub that object is no longer a dataclass, so
``load_dataset("allenai/ai2_arc", ...)`` and friends raise ``TypeError: must be
called with a dataclass type or instance`` before any data is touched.  Pulling
the repo's parquet/jsonl files by name sidesteps the metadata call entirely and
still uses the ordinary HF cache, so a compute node without network works as
long as the login node warmed the cache (``--download_only``).
"""

from __future__ import annotations

import json

from huggingface_hub import hf_hub_download

HUMANEVAL_REPO = "openai/openai_humaneval"
HUMANEVAL_FILE = "openai_humaneval/test-00000-of-00001.parquet"
MBPP_REPO = "google-research-datasets/mbpp"
MBPP_FILE = "full/test-00000-of-00001.parquet"
IFEVAL_REPO = "google/IFEval"
IFEVAL_FILE = "ifeval_input_data.jsonl"
BBH_REPO = "lukaemon/bbh"
# Math columns.  SVAMP is the 1000-problem set of Patel et al. (2021); the Hub
# copy splits it 300/700 as test/train, and the test half is what the earlier
# lm-eval phase of this project scored, so it is listed first.  AIME is the
# 2024 + 2025 exams (30 each); Minerva Math is the 272 OCW problems the
# Minerva paper released, as packaged for the Qwen2.5-Math evaluation.
SVAMP_REPO = "ChilleD/SVAMP"
SVAMP_FILES = ("data/test-00000-of-00001.parquet", "data/train-00000-of-00001.parquet")
AIME24_REPO, AIME24_FILE = "Maxwell-Jia/AIME_2024", "aime_2024_problems.parquet"
AIME25_REPO, AIME25_FILE = "math-ai/aime25", "test.jsonl"
MINERVA_REPO, MINERVA_FILE = "math-ai/minervamath", "test.jsonl"

# Multiple-choice columns (HellaSwag, ARC-C, ARC-E, PIQA).  ``load_dataset`` is
# unusable for these here for the reason in the module docstring, so each is
# pulled as the repo's own parquet file.  PIQA ships only a loading script on
# its main branch, so its rows come from the Hub's auto-converted parquet
# branch instead -- same data, no remote code and no download at eval time.
HELLASWAG_REPO, HELLASWAG_FILE = "Rowan/hellaswag", "data/validation-00000-of-00001.parquet"
ARC_REPO = "allenai/ai2_arc"
ARC_FILES = {"arc_c": "ARC-Challenge/test-00000-of-00001.parquet",
             "arc_e": "ARC-Easy/test-00000-of-00001.parquet"}
PIQA_REPO, PIQA_FILE = "ybisk/piqa", "plain_text/validation/0000.parquet"
PIQA_REVISION = "refs/convert/parquet"

# ARC-Easy is evaluated on a pre-registered 800-item subset, not the full 2365.
# It has no official LLaDA-Instruct config (the paper's Table 8 has no ARC-E
# row), so it is a mechanism diagnostic rather than a main-table column, and at
# the ARC-C protocol it costs 10-60 h per arm at full size.  800 is chosen to
# keep a paired test able to resolve a ~5pp gap (n >= ~600 at a ~0.2 discordance
# rate); the same 800 items are used by every arm.
#
# The draw is a deterministic hash order rather than random.sample, so it does
# not depend on the Python version, and it was fixed before any ARC-E answer
# was scored -- results/eval_stageB/arc_e_subset_manifest.json records both.
ARC_E_SUBSET_SEED = 20260904
ARC_E_SUBSET_N = 800

#: Option letters, shared by every multiple-choice loader below.
_LETTERS = "ABCDEFGH"

#: BBH subtask order and free-form/multiple-choice split, copied from
#: OpenCompass ``bbh_gen_ee62e9`` -- which scorer a subtask gets depends on it.
BBH_MULTIPLE_CHOICE_SETS = [
    "temporal_sequences",
    "disambiguation_qa",
    "date_understanding",
    "tracking_shuffled_objects_three_objects",
    "penguins_in_a_table",
    "geometric_shapes",
    "snarks",
    "ruin_names",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_five_objects",
    "logical_deduction_three_objects",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "salient_translation_error_detection",
    "reasoning_about_colored_objects",
]
BBH_FREE_FORM_SETS = [
    "multistep_arithmetic_two",
    "navigate",
    "dyck_languages",
    "word_sorting",
    "sports_understanding",
    "boolean_expressions",
    "object_counting",
    "formal_fallacies",
    "causal_judgement",
    "web_of_lies",
]
BBH_SETS = BBH_MULTIPLE_CHOICE_SETS + BBH_FREE_FORM_SETS


def _parquet(repo: str, filename: str, revision: str | None = None) -> list[dict]:
    import pandas as pd

    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return pd.read_parquet(path).to_dict("records")


def load_humaneval() -> list[dict]:
    """The 164 HumanEval problems, in task_id order."""
    return _parquet(HUMANEVAL_REPO, HUMANEVAL_FILE)


def load_mbpp() -> list[dict]:
    """MBPP task_id 11-510.

    OpenCompass's ``MBPPDataset`` slices the raw jsonl as ``train[10:510]``;
    the Hub's ``full`` config already ships exactly that range as its ``test``
    split (the first ten problems are the separate ``prompt`` split).
    """
    rows = _parquet(MBPP_REPO, MBPP_FILE)
    assert len(rows) == 500 and rows[0]["task_id"] == 11, "unexpected MBPP split"
    return rows


def load_ifeval() -> list[dict]:
    """The 541 IFEval prompts with their instruction ids and kwargs."""
    path = hf_hub_download(IFEVAL_REPO, IFEVAL_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_bbh(name: str) -> list[dict]:
    """One BBH subtask, as ``{'input': ..., 'target': ...}`` records."""
    return _parquet(BBH_REPO, f"{name}/test-00000-of-00001.parquet")


def _jsonl(repo: str, filename: str) -> list[dict]:
    path = hf_hub_download(repo, filename, repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_svamp() -> list[dict]:
    """All 1000 SVAMP problems as ``{'id', 'question', 'answer', 'type', 'split'}``.

    The 300-problem ``test`` half comes first, so ``--limit 300`` reproduces the
    split scored earlier in this project.  ``question`` is body + question,
    which is how SVAMP is posed; the gold answers are all integers.
    """
    rows = []
    for split, filename in zip(("test", "train"), SVAMP_FILES):
        for row in _parquet(SVAMP_REPO, filename):
            rows.append({"id": row["ID"], "split": split, "type": row["Type"],
                         "question": f"{row['Body'].strip()} {row['Question'].strip()}",
                         "answer": str(row["Answer"]).strip()})
    assert len(rows) == 1000 and rows[0]["split"] == "test", "unexpected SVAMP layout"
    return rows


def load_aime() -> list[dict]:
    """AIME 2024 (I + II) followed by AIME 2025 (I + II): 60 integer-answer problems."""
    rows = [{"id": row["ID"], "problem": row["Problem"], "answer": str(row["Answer"]).strip()}
            for row in _parquet(AIME24_REPO, AIME24_FILE)]
    rows += [{"id": f"2025-{row['id']}", "problem": row["problem"], "answer": str(row["answer"]).strip()}
             for row in _jsonl(AIME25_REPO, AIME25_FILE)]
    assert len(rows) == 60 and all(r["answer"].isdigit() for r in rows), "unexpected AIME layout"
    return rows


def load_minerva_math() -> list[dict]:
    """The 272 Minerva OCW problems; 187 have numeric answers, the rest symbolic."""
    rows = [{"id": str(i), "problem": row["question"], "answer": row["answer"].strip()}
            for i, row in enumerate(_jsonl(MINERVA_REPO, MINERVA_FILE))]
    assert len(rows) == 272, "unexpected Minerva Math layout"
    return rows


def load_hellaswag() -> list[dict]:
    """HellaSwag validation (10042 items) as ``{'id', 'context', 'options', 'gold'}``.

    ``ctx`` is the field lm-eval and OpenCompass both condition on (``ctx_a``
    plus the ``ctx_b`` sentence opener); ``label`` is the 0-based index of the
    correct ending, rendered here as its option letter.
    """
    rows = []
    for row in _parquet(HELLASWAG_REPO, HELLASWAG_FILE):
        options = [str(e).strip() for e in row["endings"]]
        rows.append({"id": str(row["ind"]), "context": str(row["ctx"]).strip(),
                     "options": options, "gold": _LETTERS[int(row["label"])]})
    assert len(rows) == 10042, "unexpected HellaSwag layout"
    return rows


def load_arc(subset: str) -> list[dict]:
    """ARC-Challenge or ARC-Easy test as ``{'id', 'question', 'options', 'gold'}``.

    Filtered exactly the way OpenCompass's ``ARCDataset`` filters it, because
    that is the dataset LLaDA's own ARC-C config evaluates: rows whose choice
    count is not 4 are dropped (7 of 1172 in Challenge, 11 of 2376 in Easy),
    and the answer key is remapped to A-D **by position**, so the handful of
    rows labelled "1".."4" line up with the rest.
    """
    rows = []
    for row in _parquet(ARC_REPO, ARC_FILES[subset]):
        options = [str(t).strip() for t in row["choices"]["text"]]
        if len(options) != 4:
            continue
        labels = [str(x) for x in row["choices"]["label"]]
        key = str(row["answerKey"])
        if key not in labels:                    # nothing to align against
            raise ValueError(f"{subset} row {row['id']}: answerKey {key!r} not in {labels}")
        rows.append({"id": str(row["id"]), "question": str(row["question"]).strip(),
                     "options": options, "gold": _LETTERS[labels.index(key)]})
    assert len(rows) == {"arc_c": 1165, "arc_e": 2365}[subset], f"unexpected {subset} layout"
    return rows


def arc_e_subset(rows: list[dict]) -> list[dict]:
    """The pre-registered ARC-Easy subset, in dataset order.

    Deterministic in the row ids alone: each id is ranked by
    ``sha256(f"{seed}:{id}")`` and the first ARC_E_SUBSET_N are kept, then put
    back into dataset order so per-item records line up across arms.
    """
    import hashlib

    def rank(row):
        payload = f"{ARC_E_SUBSET_SEED}:{row['id']}".encode()
        return hashlib.sha256(payload).hexdigest()

    keep = {row["id"] for row in sorted(rows, key=rank)[:ARC_E_SUBSET_N]}
    picked = [row for row in rows if row["id"] in keep]
    assert len(picked) == ARC_E_SUBSET_N, f"subset drew {len(picked)} rows"
    return picked


def load_piqa() -> list[dict]:
    """PIQA validation (1838 items) as ``{'id', 'goal', 'options', 'gold'}``."""
    rows = []
    for i, row in enumerate(_parquet(PIQA_REPO, PIQA_FILE, revision=PIQA_REVISION)):
        rows.append({"id": str(i), "goal": str(row["goal"]).strip(),
                     "options": [str(row["sol1"]).strip(), str(row["sol2"]).strip()],
                     "gold": _LETTERS[int(row["label"])]})
    assert len(rows) == 1838, "unexpected PIQA layout"
    return rows



def prefetch() -> None:
    """Warm the HF cache for every file these benchmarks read."""
    for repo, filename in [(HUMANEVAL_REPO, HUMANEVAL_FILE),
                           (MBPP_REPO, MBPP_FILE),
                           (IFEVAL_REPO, IFEVAL_FILE),
                           *[(SVAMP_REPO, f) for f in SVAMP_FILES],
                           (AIME24_REPO, AIME24_FILE), (AIME25_REPO, AIME25_FILE),
                           (MINERVA_REPO, MINERVA_FILE),
                           (HELLASWAG_REPO, HELLASWAG_FILE),
                           *[(ARC_REPO, f) for f in ARC_FILES.values()]]:
        hf_hub_download(repo, filename, repo_type="dataset")
        print(f"  OK  {repo}/{filename}")
    hf_hub_download(PIQA_REPO, PIQA_FILE, repo_type="dataset", revision=PIQA_REVISION)
    print(f"  OK  {PIQA_REPO}/{PIQA_FILE}")
    for name in BBH_SETS:
        hf_hub_download(BBH_REPO, f"{name}/test-00000-of-00001.parquet",
                        repo_type="dataset")
    print(f"  OK  {BBH_REPO} ({len(BBH_SETS)} subtasks)")

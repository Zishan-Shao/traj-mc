"""Convert lm-eval sample logs into paired per-item JSONL records."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re

from trajmc import common as C
from trajmc import sampling as S


def _subtask(path: str) -> str:
    return re.sub(
        r"^samples_|_\d{4}-\d{2}-\d{2}T.*$", "", os.path.basename(path)
    )


def _timestamp(path: str) -> str:
    match = re.search(
        r"samples_.*_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$",
        os.path.basename(path),
    )
    return match.group(1) if match else ""


def latest_sample_files(task_dir: str) -> list[str]:
    """Select the newest log for every leaf task, preserving MMLU shards."""
    files = glob.glob(os.path.join(task_dir, "**", "samples_*.jsonl"), recursive=True)
    newest = {}
    for path in files:
        key = _subtask(path)
        if key not in newest or _timestamp(path) > _timestamp(newest[key]):
            newest[key] = path
    return sorted(newest.values())


def metric_value(record: dict, metric: str):
    for key, value in record.items():
        if key == metric or key.startswith(metric + ","):
            return value[0] if isinstance(value, (list, tuple)) else value
    return None


def prompt_hash(record: dict) -> str:
    if "arguments" in record:
        value = record["arguments"]
    elif "doc" in record:
        value = record["doc"]
    else:
        value = record.get("doc_id")
    return C.sha256_text(json.dumps(value, sort_keys=True, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(C.BACKENDS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--arm",
        choices=sorted({"ref", "base", "ours", *S.SCHEMES}),
        required=True,
    )
    parser.add_argument("--lmeval_dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    backend = C.get_backend(args.backend)
    if args.task not in backend.tasks:
        parser.error(f"unsupported {backend.name} task: {args.task}")
    metric = C.primary_metric(backend, args.task)
    task_dir = args.lmeval_dir or str(
        C.repo_root() / "results" / "eval" / backend.name / args.arm /
        "lmeval" / args.task
    )
    files = latest_sample_files(task_dir)
    if not files:
        parser.error(f"no samples_*.jsonl found under {task_dir}")

    output = Path(args.out) if args.out else (
        C.repo_root() / "results" / "eval" / backend.name / args.arm /
        f"{C.git_hash()}_{args.arm}_{args.task}_items.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    preferred_filter = {
        ("llada", "gsm8k"): "flexible-extract",
        ("dream", "gsm8k_cot"): "flexible-extract",
    }.get((backend.name, args.task))

    count = correct = missing = 0
    with output.open("w", encoding="utf-8") as destination:
        for sample_file in files:
            subtask = _subtask(sample_file)
            with open(sample_file, encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if preferred_filter and record.get("filter") not in (
                        preferred_filter,
                        None,
                    ):
                        continue
                    value = metric_value(record, metric)
                    if value is None:
                        missing += 1
                        continue
                    is_correct = bool(round(float(value)))
                    item = {
                        "item_id": f"{subtask}-{record.get('doc_id')}",
                        "prompt_hash": prompt_hash(record),
                        "correct": is_correct,
                        "primary_metric": metric,
                        "primary_value": float(value),
                        "acc": metric_value(record, "acc"),
                        "acc_norm": metric_value(record, "acc_norm"),
                        "gold": record.get("target"),
                    }
                    destination.write(json.dumps(item) + "\n")
                    count += 1
                    correct += int(is_correct)

    summary = {
        "backend": backend.name,
        "task": args.task,
        "arm": args.arm,
        "primary_metric": metric,
        "n": count,
        "correct": correct,
        "acc_primary": correct / count if count else None,
        "missing_metric": missing,
        "n_sample_files": len(files),
        "git_hash": C.git_hash(),
        "items_file": str(output),
    }
    if output.name.endswith("_items.jsonl"):
        summary_name = output.name[:-len("_items.jsonl")] + "_summary.json"
    elif output.suffix == ".jsonl":
        summary_name = output.stem + "_summary.json"
    else:
        summary_name = output.name + "_summary.json"
    summary_path = output.with_name(summary_name)
    C.dump_json(summary, summary_path)
    print(
        f"[convert] {backend.name}/{args.arm}/{args.task}: "
        f"{correct}/{count} -> {output}"
    )


if __name__ == "__main__":
    main()

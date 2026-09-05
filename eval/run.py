"""Run the official lm-eval adapter for a dense or compressed backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from trajmc import common as C
from trajmc import sampling as S


def build_command(
    backend_name: str,
    task: str,
    arm: str,
    model_path: str,
    weights: str | None,
    num_processes: int,
    limit: str | None,
    out_root: str,
    request_batch_size: int = 4,
    lowrank_mode: str = "factorized",
):
    backend = C.get_backend(backend_name)
    config = dict(backend.tasks[task])
    harness_dir = Path(__file__).resolve().parent
    out_path = Path(out_root) / backend.name / arm / "lmeval" / task
    out_path.mkdir(parents=True, exist_ok=True)
    family = C.arch_family(backend)

    if family == "llada":
        harness = harness_dir / "llada_harness.py"
        model_name = "llada_dist"
        model_args = [
            f"model_path={model_path}",
            "is_check_greedy=False",
            f"request_batch_size={request_batch_size}",
            "fast_generate=False",
            "fast_confidence=False",
            f"lowrank_mode={lowrank_mode}",
            "lowrank_pad_to=0",
        ]
        if config["gen"]:
            model_args.extend(
                [
                    f"gen_length={config['gen_length']}",
                    f"steps={config['steps']}",
                    f"block_length={config['block_length']}",
                ]
            )
        else:
            model_args.extend(
                [f"cfg={config['cfg']}", f"mc_num={config['mc_num']}"]
            )
    else:
        harness = harness_dir / "dream_harness.py"
        model_name = "dream"
        model_args = [f"pretrained={model_path}", "add_bos_token=true"]
        if config["gen"]:
            model_args.extend(
                [
                    f"max_new_tokens={config['max_new_tokens']}",
                    f"diffusion_steps={config['diffusion_steps']}",
                    f"temperature={config['temperature']}",
                    f"top_p={config['top_p']}",
                ]
            )

    if weights:
        model_args.append(f"weights_path={weights}")
        if family == "dream":
            model_args.append(f"lowrank_mode={lowrank_mode}")

    port = 20000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 20000)
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_processes),
        "--main_process_port",
        str(port),
        str(harness),
        "--model",
        model_name,
        "--tasks",
        str(config.get("lm_eval_task", task)),
        "--num_fewshot",
        str(config["fs"]),
        "--batch_size",
        str(config["batch_size"]),
        "--model_args",
        ",".join(model_args),
        "--output_path",
        str(out_path),
        "--log_samples",
    ]
    if family == "dream":
        command.append("--confirm_run_unsafe_code")
    if limit:
        command.extend(["--limit", str(limit)])
    return command, out_path, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(C.BACKENDS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--arm",
        choices=sorted({"ref", "base", "ours", *S.SCHEMES}),
        required=True,
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--request_batch_size", type=int, default=4)
    parser.add_argument("--lowrank_mode", choices=["factorized", "materialized"],
                        default="factorized")
    parser.add_argument("--limit", default=None)
    parser.add_argument("--out_root", default=str(C.repo_root() / "results" / "eval"))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    backend = C.get_backend(args.backend)
    if args.task not in backend.tasks:
        choices = ", ".join(backend.tasks)
        parser.error(f"unsupported {backend.name} task {args.task!r}; choose: {choices}")
    if args.arm == "ref" and args.weights:
        parser.error("the dense ref arm must not receive --weights")
    if args.arm != "ref" and not args.weights:
        parser.error(f"the {args.arm} arm requires --weights")

    model_path = args.model_path or backend.model_id
    weights = str(Path(args.weights).expanduser().resolve()) if args.weights else None
    if weights:
        summary_path = Path(weights) / "compression_summary.json"
        if not summary_path.is_file():
            parser.error(f"missing compression summary: {summary_path}")
        summary = C.load_json(summary_path)
        if summary.get("backend") not in (None, backend.name):
            parser.error(
                f"weights backend={summary.get('backend')!r} does not match "
                f"--backend={backend.name!r}"
            )

    command, out_path, task_config = build_command(
        backend.name,
        args.task,
        args.arm,
        model_path,
        weights,
        args.num_processes,
        args.limit,
        args.out_root,
        args.request_batch_size,
        args.lowrank_mode,
    )
    print("[evaluate] " + " ".join(command))
    if args.dry:
        return

    manifest = {
        "backend": backend.name,
        "task": args.task,
        "task_config": task_config,
        "arm": args.arm,
        "model_path": model_path,
        "weights_path": weights,
        "command": command,
        "git_hash": C.git_hash(),
        "started_unix": time.time(),
    }
    manifest_path = out_path / "trajmc_eval_manifest.json"
    C.dump_json(manifest, manifest_path)
    env = dict(os.environ)
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    result = subprocess.run(command, env=env, check=False)
    manifest.update(exit_code=result.returncode, finished_unix=time.time())
    C.dump_json(manifest, manifest_path)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

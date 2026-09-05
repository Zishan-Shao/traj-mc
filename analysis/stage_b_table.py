"""Assemble the Stage B main table from whatever has finished so far.

The default columns are the plan's main table: Gen. Recon. plus GSM8K,
MATH-500 and MMLU.  ``--benches`` swaps in any other set the harness has run
(humaneval, mbpp, ifeval, bbh) for the wider capability table.  Cells that have
not finished are left blank rather than omitted, so the table doubles as a
progress sheet.
"""

from __future__ import annotations

import argparse
import json
import os

ARMS = [("weight_svd", "Weight SVD"), ("clean", "Clean-SVD"),
        ("mcs", "MCS-SVD"), ("traj", "Traj-SVD"),
        ("actual", "Actual-State (control)")]
ALL_BENCHES = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("mmlu", "MMLU"),
               ("svamp", "SVAMP"), ("aime", "AIME"), ("minerva_math", "Minerva"),
               ("arc_c", "ARC-C"), ("arc_e", "ARC-E"),
               ("hellaswag", "HellaSwag"), ("piqa", "PIQA"),
               ("humaneval", "HumanEval"), ("mbpp", "MBPP"),
               ("ifeval", "IFEval"), ("bbh", "BBH")]
DEFAULT_BENCHES = ["gsm8k", "math500", "mmlu"]
#: The capability table the plan asks for: the three finished columns plus the
#: five queued in this round.  `--benches wide` expands to exactly this.
WIDE_BENCHES = ["gsm8k", "math500", "mmlu", "svamp",
                "arc_c", "arc_e", "hellaswag", "piqa"]
RATIOS = [("r08", "0.8", "20%"), ("r06", "0.6", "40%")]


def read_eval(cell: str, bench: str, root: str):
    # The harness also drops a "<bench>_ckpt.json" checkpoint mid-run; only the
    # final file carries the run's metadata, so match it exactly.
    path = os.path.join(root, cell, f"{bench}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        blob = json.load(handle)
    entry = blob.get("results", {}).get(bench)
    if entry is None:                      # harness keys MATH-500 under 'math500'
        entry = next(iter(blob.get("results", {}).values()), None)
    if not isinstance(entry, dict) or entry.get("acc") is None:
        return None
    return entry.get("acc"), entry.get("n")


def read_recon(ratio_dir: str, mode: str):
    path = os.path.join(ratio_dir, "reports",
                        f"gen_reconstruction_{os.path.basename(ratio_dir).split('_')[-1]}_{mode}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        blob = json.load(handle)
    return {k: v["E_gen"] for k, v in blob.get("arms", {}).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="results/eval_stageB")
    parser.add_argument("--stage_root", default="results/stage_a")
    parser.add_argument("--model", default="llada_instruct")
    parser.add_argument("--positions", choices=["generation", "all"],
                        default="generation")
    parser.add_argument("--benches", default=",".join(DEFAULT_BENCHES),
                        help="comma-separated column set, or 'wide' for "
                             + ",".join(WIDE_BENCHES) + "; choose from "
                             + ", ".join(name for name, _ in ALL_BENCHES))
    args = parser.parse_args()

    labels = dict(ALL_BENCHES)
    if args.benches.strip() == "wide":
        args.benches = ",".join(WIDE_BENCHES)
    wanted = [name.strip() for name in args.benches.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in labels]
    if unknown:
        parser.error(f"unknown benchmark(s): {', '.join(unknown)}")
    benches = [(name, labels[name]) for name in wanted]

    dense = {b: read_eval("dense", b, args.eval_root) for b, _ in benches}
    print(f"# Stage B main table — {args.model}, E_gen on {args.positions} positions\n")
    head = f"| {'Model':<14} | {'Reduction':<9} | {'Method':<22} | {'Gen.Recon.':>10} |"
    for _, label in benches:
        head += f" {label:>9} |"
    print(head)
    widths = [14, 9, 22, 10] + [9] * len(benches)
    print("|" + "|".join(["-" * (w + 2) for w in widths]) + "|")

    row = f"| {'LLaDA-Inst':<14} | {'—':<9} | {'Dense BF16':<22} | {'—':>10} |"
    for b, _ in benches:
        v = dense.get(b)
        row += f" {(f'{v[0]*100:.1f}' if v else '—'):>9} |"
    print(row)

    for tag, ratio, reduction in RATIOS:
        recon = read_recon(os.path.join(args.stage_root, f"{args.model}_r{ratio}"),
                           args.positions)
        for arm, label in ARMS:
            cell = f"{tag}_{arm}"
            e = recon.get(arm)
            row = (f"| {'LLaDA-Inst':<14} | {reduction:<9} | {label:<22} | "
                   f"{(f'{e:.5f}' if e is not None else '—'):>10} |")
            for b, _ in benches:
                v = read_eval(cell, b, args.eval_root)
                row += f" {(f'{v[0]*100:.1f}' if v else '—'):>9} |"
            print(row)

    n_cells = (len(RATIOS) * len(ARMS) + 1) * len(benches)
    total = sum(1 for t, _, _ in RATIOS for a, _ in ARMS for b, _ in benches
                if read_eval(f"{t}_{a}", b, args.eval_root)) \
        + sum(1 for b, _ in benches if dense.get(b))
    print(f"\n完成 {total}/{n_cells} 格")


if __name__ == "__main__":
    main()

"""Print every number in the paper's Table 1 and Figs. 1b, 2a and 2b from the
outputs of the scripts in scripts/.  Missing cells are shown as '-', so this
also works as a progress sheet.

    python utils/collect_results.py [--results results]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

BENCHES = ["GSM8K", "MATH-500", "SVAMP", "ARC-C", "MMLU", "HellaSwag", "PIQA"]

LLADA_KEYS = ["gsm8k", "math500", "svamp", "arc_c", "mmlu", "hellaswag", "piqa"]
LLADA_ROWS = [("Dense", "dense"),
              ("20% SVD", "r0.8_weight_svd"), ("20% SVD-LLM", "r0.8_clean"),
              ("20% +Traj-MC", "r0.8_traj"),
              ("40% SVD", "r0.6_weight_svd"), ("40% SVD-LLM", "r0.6_clean"),
              ("40% +Traj-MC", "r0.6_traj")]

# (lm-eval task, metric) per column, in BENCHES order.
DREAM_KEYS = [("gsm8k_cot", "exact_match,flexible-extract"),
              ("minerva_math500", "math_verify,none"),
              ("svamp_cot", "exact_match,flexible-extract"),
              ("arc_challenge", "acc,none"),
              ("mmlu_generative", "exact_match,get_response"),
              ("hellaswag", "acc,none"),
              ("piqa", "acc,none")]
DREAM_ROWS = [("Dense", "dense"), ("20% SVD", "nokv_weight_svd"),
              ("20% SVD-LLM", "nokv_clean"), ("20% +Traj-MC", "nokv_traj")]


def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def llada_cell(root, cell, bench):
    blob = load(os.path.join(root, "llada_instruct", "eval", cell, f"{bench}.json"))
    entry = (blob or {}).get("results", {}).get(bench)
    return None if not entry or entry.get("acc") is None else 100 * entry["acc"]


def dream_cell(root, cell, task, metric):
    files = sorted(glob.glob(os.path.join(
        root, "dream_instruct", "eval", cell, task, "*", "results_*.json")))
    if not files:
        return None
    results = load(files[-1])["results"]
    entry = results.get(task) or next(iter(results.values()))
    value = entry.get(metric)
    return None if value is None else 100 * value


def table(title, rows):
    print(f"\n## {title}\n")
    print("| Setting | " + " | ".join(BENCHES) + " |")
    print("|---|" + "---:|" * len(BENCHES))
    for label, values in rows:
        cells = ["-" if v is None else f"{v:.1f}" for v in values]
        print(f"| {label} | " + " | ".join(cells) + " |")


def egen(path, title):
    blob = load(path)
    print(f"\n## {title}\n")
    if not blob:
        print(f"(missing {path})")
        return
    for arm, row in blob["arms"].items():
        print(f"  {arm:14s} E_gen = {row['E_gen']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    root = parser.parse_args().results
    reports = os.path.join(root, "llada_instruct", "reports")

    table("Table 1 -- LLaDA-8B-Instruct",
          [(label, [llada_cell(root, cell, b) for b in LLADA_KEYS])
           for label, cell in LLADA_ROWS])
    table("Table 1 -- Dream-7B-Instruct (k/v projections kept dense)",
          [(label, [dream_cell(root, cell, t, m) for t, m in DREAM_KEYS])
           for label, cell in DREAM_ROWS])

    egen(os.path.join(reports, "gen_reconstruction_r0.8_generation.json"),
         "Fig. 1b -- E_gen on held-out generation states (20%)")
    egen(os.path.join(reports, "fixed_t_r0.8_generation.json"),
         "Fig. 2a -- fixed t vs t ~ U(0,1)")

    print("\n## Fig. 2b -- relative Frobenius error of Sigma_N\n")
    blob = load(os.path.join(reports, "mc_budget_covariance.json"))
    if not blob:
        print("(missing mc_budget_covariance.json)")
    else:
        for label, row in blob["comparisons"].items():
            if label == blob["reference"]:
                continue
            err = row["aggregate"]["mean_relative_frobenius_error"]
            print(f"  {label:16s} e_Sigma = {err:.4f}")


if __name__ == "__main__":
    main()

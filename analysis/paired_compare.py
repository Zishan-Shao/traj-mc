"""Paired comparison between two Stage B evaluation cells.

The two arms answer the same questions, so the comparison is paired: McNemar
on the discordant pairs plus a paired bootstrap over questions, as the plan
requires. An unpaired difference of proportions would ignore that pairing and
overstate the uncertainty.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import binomtest


def load(cell: str, bench: str, root: str) -> dict[int, bool]:
    path = os.path.join(root, cell, f"{bench}.json")
    blob = json.load(open(path))
    per_item = blob["per_item"]
    recs = (per_item[bench] if isinstance(per_item, dict) and bench in per_item
            else (list(per_item.values())[0] if isinstance(per_item, dict)
                  else per_item))
    return {int(r["idx"]): bool(r["correct"]) for r in recs}


def compare(a: dict, b: dict, label_a: str, label_b: str, draws: int, seed: int):
    common = sorted(set(a) & set(b))
    ca = np.array([a[i] for i in common])
    cb = np.array([b[i] for i in common])
    n01 = int((~ca & cb).sum())      # b right, a wrong
    n10 = int((ca & ~cb).sum())
    mcnemar = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), (draws, len(common)))
    diff = cb[idx].mean(axis=1) - ca[idx].mean(axis=1)
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return {
        "n": len(common),
        f"acc_{label_a}": float(ca.mean()),
        f"acc_{label_b}": float(cb.mean()),
        "delta": float(cb.mean() - ca.mean()),
        "ci95": [float(lo), float(hi)],
        "mcnemar_p": float(mcnemar),
        "discordant": {f"{label_b}_only": n01, f"{label_a}_only": n10},
        "significant": bool(lo > 0 or hi < 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--root", default="results/eval_stageB")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = compare(load(args.baseline, args.bench, args.root),
                  load(args.arm, args.bench, args.root),
                  args.baseline, args.arm, args.draws, args.seed)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

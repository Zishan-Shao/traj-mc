"""Read out the MATH-500 sampling-family pilot and apply the pre-registered rule.

LLaDA publishes MATH-500 for LLaDA-8B-Instruct under two official samplers:

  pure diffusion   gen 512 / block 512, confidence_eos_eot_inf=True   -> 29.6
  block diffusion  gen 512 / block  64, both eos flags False          -> 42.7

The finished column ran the first.  The question is whether that choice changes
the conclusion the column carries, so the pilot compares the two families on
one fixed 100-problem prefix across three arms.

The pure-diffusion side costs nothing: those 100 problems are the first 100 of
the finished full runs, so they are sliced out of their per-item records rather
than regenerated.  Only the block-diffusion side needed GPUs.

The rule below was fixed before any block-diffusion number existed, and is
recorded with its amendment in results/pilot_math500/items_manifest.json:

  keep   block-side Traj - Clean > 0, or <= 0 but not paired-significant
  rerun  block-side Traj - Clean <= 0 AND paired-significant (McNemar p < 0.05
         or the one-sided bootstrap CI excludes 0)

The original rule tested the bare sign of the gap. It was amended, before any
block-diffusion answer was generated, because SE(gap) at n=300 is about 2.2pp
-- the same order as the gap itself (~2pp on the full 500) -- so the sign is
noise-dominated at any n reachable here and a bare sign test would trigger a
~117 GPU-h column rerun on noise. The answerable question is "significant
reversal", not "sign".

A higher block-diffusion score is explicitly NOT a reason to switch.
"""

from __future__ import annotations

import argparse
import json
import os

ARMS = [("dense", "Dense"), ("r08_clean", "Clean-SVD 20%"), ("r08_traj", "Traj-SVD 20%")]
OFFICIAL = {"pure": 29.6, "block": 42.7}
N_PILOT = 100


def pure_side(arm: str, root: str, n: int):
    """The pilot's items sliced out of the finished full run -- no new compute.

    eval_math500 walks the unshuffled test set in order and stores each answer's
    position as `idx`, and the pilot's items are that set's first `n`, so
    `idx < n` is exactly the pilot slice.
    """
    path = os.path.join(root, arm, "math500.json")
    if not os.path.exists(path):
        return None
    blob = json.load(open(path))
    per_item = blob.get("per_item") or {}
    recs = per_item.get("math500") if isinstance(per_item, dict) else per_item
    if not recs:
        return None
    sliced = sorted((r for r in recs if n is None or int(r["idx"]) < n),
                    key=lambda r: int(r["idx"]))
    if n is not None and len(sliced) != n:
        return None
    n = len(sliced)
    return {"acc": 100 * sum(bool(r["correct"]) for r in sliced) / n,
            "n": n, "correct": [bool(r["correct"]) for r in sliced],
            "full_acc": 100 * blob["results"]["math500"]["acc"],
            "sampler": blob.get("bench_config", {}).get("math500")}


def block_side(arm: str, root: str):
    path = os.path.join(root, f"{arm}_blk.json")
    if not os.path.exists(path):
        return None
    blob = json.load(open(path))
    entry = blob.get("results", {}).get("math500")
    if not entry or entry.get("acc") is None:
        return None
    per_item = blob.get("per_item") or {}
    recs = per_item.get("math500") if isinstance(per_item, dict) else per_item
    recs = sorted(recs or [], key=lambda r: int(r["idx"]))
    return {"acc": 100 * entry["acc"], "n": entry["n"],
            "correct": [bool(r["correct"]) for r in recs],
            "diag": blob.get("gen_diag"),
            "sampler": blob.get("bench_config", {}).get("math500")}


def paired(a, b, draws=20000, seed=0):
    """Paired gap b - a in pp, its bootstrap CI, and the McNemar p.

    Same machinery as analysis/paired_compare.py: the two arms answer the same
    questions, so an unpaired test would throw away the pairing and overstate
    the uncertainty.
    """
    import numpy as np
    from scipy.stats import binomtest
    ca, cb = np.array(a, dtype=bool), np.array(b, dtype=bool)
    n01 = int((~ca & cb).sum())
    n10 = int((ca & ~cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ca), (draws, len(ca)))
    diff = 100 * (cb[idx].mean(axis=1) - ca[idx].mean(axis=1))
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return {"gap": 100 * (cb.mean() - ca.mean()), "lo": float(lo), "hi": float(hi),
            "p": float(p), "significant": bool(p < 0.05 or lo > 0 or hi < 0),
            "discordant": (n01, n10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pilot_math500")
    ap.add_argument("--eval_root", default="results/eval_stageB")
    ap.add_argument("--samples", type=int, default=0,
                    help="print this many decoded block-diffusion answers per arm")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.root, "items_manifest.json")))
    n = manifest["n"]
    print(f"# MATH-500 sampling-family pilot — {n} fixed items "
          f"(ids sha256 {manifest['ids_sha256'][:16]})\n")

    pure = {a: pure_side(a, args.eval_root, n) for a, _ in ARMS}
    block = {a: block_side(a, args.root) for a, _ in ARMS}

    def cell(side, key, width, pct=True):
        if side is None or side.get(key) is None:
            return f"{'—':>{width}}"
        return f"{side[key]:{width - 1}.1f}%" if pct else f"{side[key]:>{width}}"

    print(f"| {'Arm':<14} | {'pure (block 512)':>17} | {'block (block 64)':>17} | "
          f"{'full-500 pure':>13} |")
    print("|" + "|".join("-" * w for w in (16, 19, 19, 15)) + "|")
    for arm, label in ARMS:
        print(f"| {label:<14} | {cell(pure[arm], 'acc', 17)} | "
              f"{cell(block[arm], 'acc', 17)} | {cell(pure[arm], 'full_acc', 13)} |")
    print(f"\n  official OpenCompass: pure {OFFICIAL['pure']}, block {OFFICIAL['block']}")

    print("\n## Collapse check (block-diffusion side)")
    for arm, label in ARMS:
        b = block[arm]
        if not b or not b.get("diag"):
            print(f"  {label:<14} —")
            continue
        d = b["diag"]
        print(f"  {label:<14} mean pad {d['mean_pad_frac']:.3f}  "
              f"mean distinct {d['mean_distinct']:.1f}  "
              f">90% pad {d['frac_answers_over_90pct_pad']:.0%}  "
              f"empty {d['frac_answers_empty']:.0%}")

    print("\n## Decision")
    stats = {}
    for fam, side in (("pure", pure), ("block", block)):
        c, t = side["r08_clean"], side["r08_traj"]
        if not c or not t:
            print(f"  {fam:<6}: incomplete")
            continue
        st = paired(c["correct"], t["correct"])
        stats[fam] = st
        print(f"  {fam:<6}: Clean {c['acc']:.1f}%  Traj {t['acc']:.1f}%   "
              f"gap {st['gap']:+.1f}pp  CI95 [{st['lo']:+.1f}, {st['hi']:+.1f}]  "
              f"McNemar p={st['p']:.3f}  ({'significant' if st['significant'] else 'n.s.'})")

    print(f"\n  full-500 pure-diffusion anchor (already on disk, no pilot cost):")
    fc, ft = pure_side("r08_clean", args.eval_root, None), pure_side("r08_traj", args.eval_root, None)
    if fc and ft:
        st = paired(fc["correct"], ft["correct"])
        print(f"    Clean {fc['acc']:.1f}%  Traj {ft['acc']:.1f}%   gap {st['gap']:+.1f}pp  "
              f"CI95 [{st['lo']:+.1f}, {st['hi']:+.1f}]  McNemar p={st['p']:.3f}  "
              f"({'significant' if st['significant'] else 'n.s.'}, n={len(fc['correct'])})")
        print("    MATH-500 sits near the floor for the compressed arms, so this column is "
              "not\n    independently significant; its role is supporting evidence whose "
              "direction agrees\n    with GSM8K (+14.5pp, p=5e-21). That also lowers what "
              "the family choice can cost.")

    if "block" not in stats:
        print("\n  → not decidable yet")
        return
    b = stats["block"]
    if b["gap"] > 0:
        print(f"\n  → KEEP the finished pure-diffusion column: block-side gap "
              f"{b['gap']:+.1f}pp still favours Traj")
    elif not b["significant"]:
        print(f"\n  → KEEP the finished pure-diffusion column: block-side gap "
              f"{b['gap']:+.1f}pp is negative but not paired-significant "
              f"(p={b['p']:.3f}); recorded as 'the two families show no significant "
              f"difference in the Clean/Traj ordering'")
    else:
        print(f"\n  → RERUN the MATH-500 column under block diffusion: block-side gap "
              f"{b['gap']:+.1f}pp is negative and paired-significant (p={b['p']:.3f}, "
              f"CI95 [{b['lo']:+.1f}, {b['hi']:+.1f}])")
    print("\n     table note if kept: MATH-500 is the official pure-diffusion setting "
          "(gen 512 /\n     block 512), GSM8K the official block-diffusion one (gen 256 / "
          "block 8); the pilot\n     found no significant ordering difference between the "
          "two families.")

    if args.samples:
        print("\n## Decoded block-diffusion answers")
        for arm, label in ARMS:
            b = block[arm]
            if not b or not b.get("diag"):
                continue
            print(f"\n--- {label} ---")
            for i, text in enumerate(b["diag"]["first_10_answers"][:args.samples]):
                print(f"[{i}] {text[:400]}")


if __name__ == "__main__":
    main()

"""
mcnemar.py -- paired McNemar test for BASE vs OURS on a single benchmark.

Reads two per-item JSONL files (base, ours), pairs by item_id, and computes:
  b = #(BASE wrong, OURS correct)
  c = #(BASE correct, OURS wrong)
  chi2 = (b - c)^2 / (b + c)          [the pre-registered statistic]
  p_chi2  : chi-square (df=1) p-value
  p_exact : exact binomial McNemar p (recommended when b+c is small)

Reports raw acc for both arms + (b, c) + paired p. Non-paired z tests are FORBIDDEN.

Self-test (`python mcnemar.py --self_test`) validates p_chi2 and p_exact against
scipy (chi2.sf and binomtest / statsmodels.mcnemar).
"""
import sys
import json
import argparse

from .. import common as C


def _chi2_sf_df1(x):
    """Survival function of chi-square df=1 = erfc(sqrt(x/2)). No scipy needed."""
    import math

    return math.erfc(math.sqrt(x / 2.0))


def _exact_binom_two_sided(b, c):
    """
    Exact two-sided McNemar p = P(X<=min | n=b+c, 0.5) tail, doubled/clamped.
    Computed in log space (lgamma) so it is stable for large n -- the naive
    math.comb(n,i)*0.5**n overflows float for n in the thousands (e.g. MMLU).
    """
    import math

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    ln_half = math.log(0.5)
    ln_nfac = math.lgamma(n + 1)
    cum = 0.0
    for i in range(0, k + 1):
        logpmf = ln_nfac - math.lgamma(i + 1) - math.lgamma(n - i + 1) + n * ln_half
        cum += math.exp(logpmf)
    return min(1.0, 2.0 * cum)


def mcnemar_from_pairs(base_correct, ours_correct):
    assert len(base_correct) == len(ours_correct)
    b = sum(1 for bb, oo in zip(base_correct, ours_correct) if (not bb) and oo)
    c = sum(1 for bb, oo in zip(base_correct, ours_correct) if bb and (not oo))
    n = len(base_correct)
    chi2 = ((b - c) ** 2) / (b + c) if (b + c) > 0 else 0.0
    return {
        "n_paired": n,
        "acc_base": sum(base_correct) / n if n else None,
        "acc_ours": sum(ours_correct) / n if n else None,
        "b_base_wrong_ours_right": b,
        "c_base_right_ours_wrong": c,
        "chi2": chi2,
        "p_chi2": _chi2_sf_df1(chi2) if (b + c) > 0 else 1.0,
        "p_exact": _exact_binom_two_sided(b, c),
        "direction": "ours>base" if b > c else ("base>ours" if c > b else "tie"),
    }


def read_items(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["item_id"]] = r
    return d


def pair_and_test(base_path, ours_path, field="correct"):
    base = read_items(base_path)
    ours = read_items(ours_path)
    common_ids = sorted(set(base) & set(ours))
    if len(common_ids) != len(base) or len(common_ids) != len(ours):
        print(f"[mcnemar] WARNING: base={len(base)} ours={len(ours)} "
              f"paired={len(common_ids)}; using intersection.")
    # prompt_hash sanity: same item must be the same prompt across arms
    mism = [i for i in common_ids
            if base[i].get("prompt_hash") != ours[i].get("prompt_hash")]
    if mism:
        print(f"[mcnemar] WARNING: {len(mism)} items have mismatched prompt_hash "
              f"(base vs ours); protocol drift?")
    bc = [bool(base[i][field]) for i in common_ids]
    oc = [bool(ours[i][field]) for i in common_ids]
    res = mcnemar_from_pairs(bc, oc)
    res["prompt_hash_mismatches"] = len(mism)
    res["field"] = field
    return res


def self_test():
    import random

    random.seed(0)
    bc = [random.random() < 0.6 for _ in range(500)]
    oc = [random.random() < 0.65 for _ in range(500)]
    res = mcnemar_from_pairs(bc, oc)
    b, c = res["b_base_wrong_ours_right"], res["c_base_right_ours_wrong"]
    ok = True
    try:
        from scipy.stats import chi2 as sp_chi2, binomtest

        sp_p_chi2 = float(sp_chi2.sf(res["chi2"], 1))
        sp_p_exact = float(binomtest(min(b, c), b + c, 0.5).pvalue)
        d_chi2 = abs(sp_p_chi2 - res["p_chi2"])
        d_exact = abs(sp_p_exact - res["p_exact"])
        print(f"[self_test] b={b} c={c} chi2={res['chi2']:.4f}")
        print(f"[self_test] p_chi2 ours={res['p_chi2']:.6e} scipy={sp_p_chi2:.6e} "
              f"|d|={d_chi2:.2e}")
        print(f"[self_test] p_exact ours={res['p_exact']:.6e} scipy={sp_p_exact:.6e} "
              f"|d|={d_exact:.2e}")
        ok = d_chi2 < 1e-9 and d_exact < 1e-9
    except ImportError:
        print("[self_test] scipy not available; statsmodels fallback")
        try:
            from statsmodels.stats.contingency_tables import mcnemar as sm_mc
            import numpy as np

            table = np.array([[0, b], [c, 0]])
            sm = sm_mc(table, exact=True)
            d = abs(float(sm.pvalue) - res["p_exact"])
            print(f"[self_test] p_exact ours={res['p_exact']:.6e} "
                  f"statsmodels={float(sm.pvalue):.6e} |d|={d:.2e}")
            ok = d < 1e-9
        except ImportError:
            print("[self_test] neither scipy nor statsmodels; cannot validate")
            ok = False
    print(f"[self_test] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_items")
    ap.add_argument("--ours_items")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--field", default="correct", choices=["correct", "correct_norm"],
                    help="acc (correct) vs acc_norm (correct_norm)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    res = pair_and_test(args.base_items, args.ours_items, args.field)
    res["benchmark"] = args.benchmark
    print(json.dumps(res, indent=2))
    if args.out:
        C.dump_json(res, args.out)


if __name__ == "__main__":
    main()

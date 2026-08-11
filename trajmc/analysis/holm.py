"""
holm.py -- Holm-Bonferroni correction across a benchmark family.

The per-benchmark McNemar p-values are corrected as one family. Reports raw and
corrected p plus a conservative read-out:
  significant = Holm-corrected p < 0.05.

Input: a JSON mapping {benchmark: p_value} (or a directory of mcnemar outputs).
"""
import os
import json
import argparse

from .. import common as C


def holm_bonferroni(pvals, alpha=0.05):
    """
    pvals: dict{name: p}. Returns dict{name: {p, p_holm, significant}}.
    Standard step-down Holm with monotone enforcement of adjusted p-values.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for rank, (name, p) in enumerate(items):
        adj = (m - rank) * p
        adj = max(prev, adj)          # enforce monotonicity (step-down)
        adj = min(1.0, adj)
        prev = adj
        out[name] = {"p": p, "p_holm": adj, "significant": adj < alpha,
                     "rank": rank + 1}
    return out


def read_pvals(path, which="p_chi2"):
    """Load p-values from a dir of *mcnemar*.json or a single {bench:p} json."""
    if os.path.isdir(path):
        pvals = {}
        for fn in os.listdir(path):
            if fn.endswith(".json") and "mcnemar" in fn:
                d = C.load_json(os.path.join(path, fn))
                name = d.get("benchmark") or fn
                pvals[name] = d[which]
        return pvals
    d = C.load_json(path)
    if all(isinstance(v, (int, float)) for v in d.values()):
        return d
    return {k: v[which] for k, v in d.items()}


def readout(corrected):
    """Pre-registered interpretation (README section 8)."""
    n_sig = sum(1 for v in corrected.values() if v["significant"])
    dirs = {}  # requires direction info; caller may merge
    lines = []
    for name, v in sorted(corrected.items()):
        lines.append(f"  {name:14s} p={v['p']:.4g}  p_holm={v['p_holm']:.4g}  "
                     f"{'SIG' if v['significant'] else 'ns'}")
    verdict = ("claim1_supported (>=2 corrected-significant; check same-direction "
               "majority upstream)" if n_sig >= 2 else
               "mixed_or_single (report as mixed, do not overstate)")
    return n_sig, verdict, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvals", required=True, help="dir of mcnemar json OR {bench:p} json")
    ap.add_argument("--which", default="p_chi2", choices=["p_chi2", "p_exact"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pvals = read_pvals(args.pvals, args.which)
    corrected = holm_bonferroni(pvals, args.alpha)
    n_sig, verdict, table = readout(corrected)
    print(f"Holm-Bonferroni ({args.which}, alpha={args.alpha}, m={len(pvals)}):")
    print(table)
    print(f"n_significant={n_sig}  verdict={verdict}")
    result = {"which": args.which, "alpha": args.alpha, "corrected": corrected,
              "n_significant": n_sig, "verdict": verdict}
    if args.out:
        C.dump_json(result, args.out)


if __name__ == "__main__":
    main()

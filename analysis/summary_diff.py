"""
summary_diff.py -- HARD GATE: BASE vs OURS compression_summary.json must be
identical except calibration/factor identity and runtime fields. If anything
else differs (replaced-layer count, per-layer rank, parameter counts,
kept_fraction, decomposition, backend, or model), the two arms are not a clean
single-variable comparison and MUST NOT proceed to eval.

Exit 0 means the gate passes; exit 1 means it fails.

Fields ALLOWED to differ are calibration/factor identity and runtime metadata.
The model, backend, target layers, ranks, compression math, and retained
parameter count must still match exactly.
Everything else (incl. the full per_layer rank map) must match byte-for-byte.
"""
import os
import sys
import argparse

from trajmc import common as C

ALLOWED_DIFFER = {
    "arm",
    "calib_file",
    "calib_sha256",
    "calib_scheme",
    "calib_objective",
    "peak_rss_gb",
    "run_id",
    "save_path",
    "wall_clock_sec",
    "weights_sha256",
}


def _load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "compression_summary.json")
    return C.load_json(path), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="BASE weights dir or summary.json")
    ap.add_argument("--ours", required=True, help="OURS weights dir or summary.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b, bp = _load(args.base)
    o, op = _load(args.ours)
    keys = (set(b) | set(o)) - ALLOWED_DIFFER
    mism = []
    for k in sorted(keys):
        if b.get(k) != o.get(k):
            # summarize per_layer mismatch compactly
            if k == "per_layer":
                bl, ol = b.get(k, {}), o.get(k, {})
                diff_layers = [n for n in (set(bl) | set(ol)) if bl.get(n) != ol.get(n)]
                mism.append(("per_layer", f"{len(diff_layers)} layers differ: "
                                          f"{diff_layers[:5]}{'...' if len(diff_layers)>5 else ''}"))
            else:
                mism.append((k, f"base={b.get(k)!r} ours={o.get(k)!r}"))

    result = {
        "base_summary": bp, "ours_summary": op,
        "allowed_differ": sorted(ALLOWED_DIFFER),
        "n_mismatch": len(mism),
        "mismatches": {k: v for k, v in mism},
        "shared": {"ratio": b.get("ratio"), "layer_type": b.get("layer_type"),
                   "decomp": b.get("decomp"), "model_id": b.get("model_id"),
                   "n_compressed": b.get("n_compressed"),
                   "kept_fraction_base": b.get("kept_fraction"),
                   "kept_fraction_ours": o.get("kept_fraction"),
                   "git_hash": b.get("git_hash")},
        "gate": "PASS" if not mism else "FAIL",
    }
    if args.out:
        C.dump_json(result, args.out)
    else:
        C.dump_json(
            result,
            C.repo_root() / "results" / "compress" /
            f"{C.git_hash()}_summary_diff_{b.get('layer_type')}_r{b.get('ratio')}.json",
        )
    if mism:
        print(f"[summary_diff] GATE FAIL: {len(mism)} field(s) differ beyond noise:")
        for k, v in mism:
            print(f"    {k}: {v}")
        sys.exit(1)
    print(f"[summary_diff] GATE PASS: BASE/OURS identical except {sorted(ALLOWED_DIFFER)}")
    print(f"    ratio={b.get('ratio')} layer_type={b.get('layer_type')} "
          f"n_compressed={b.get('n_compressed')} "
          f"kept_fraction base={b.get('kept_fraction'):.4f} ours={o.get('kept_fraction'):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()

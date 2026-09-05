"""Mark degenerate Actual-State rollout artifacts as void.

A rollout whose sampler flooded the generation region with one token is not a
usable control: the covariance it yields describes the model reading EOS
padding, not the model generating.  Any downstream reference subspace or
GapClosed denominator computed from it is meaningless.

This tool re-runs the degeneracy diagnostics on an existing artifact, records
the verdict and the reason in its manifest, and renames the tensor so it cannot
be picked up by accident.  It never deletes anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajmc import common as C
from trajmc import sampling as S

VOID_SUFFIX = ".VOID"


def manifest_for(calib_path: Path) -> Path:
    return Path(str(calib_path).replace("_calib.pt", "_manifest.json"))


def inspect(calib_path: Path, eos_token_ids: list[int] | None) -> dict:
    blob = torch.load(calib_path, map_location="cpu", weights_only=False)
    if blob.get("scheme") != "rollout":
        raise ValueError(f"{calib_path.name} is not a rollout artifact")
    diagnostics = S.revealed_token_diagnostics(
        blob["input_ids"],
        int(blob["mask_id"]),
        int(blob.get("prefix_length") or 0),
        eos_token_ids=eos_token_ids,
    )
    try:
        S.assert_rollout_not_degenerate(diagnostics)
        return {"degenerate": False, "reason": None, **diagnostics}
    except RuntimeError as failure:
        return {"degenerate": True, "reason": str(failure), **diagnostics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calib", nargs="+", help="rollout *_calib.pt paths")
    parser.add_argument("--eos_token_id", type=int, action="append", default=[])
    parser.add_argument("--note", default=None,
                        help="extra explanation stored in the manifest")
    parser.add_argument("--apply", action="store_true",
                        help="write the manifest verdict and rename the tensor; "
                             "without it, only report")
    args = parser.parse_args()

    voided, kept = [], []
    for raw in args.calib:
        path = Path(raw).expanduser().resolve()
        verdict = inspect(path, args.eos_token_id or None)
        label = "DEGENERATE" if verdict["degenerate"] else "ok"
        print(f"[void] {label:11s} {path.name}")
        print(f"[void]   unique={verdict['unique_revealed_tokens']} "
              f"ratio={verdict['unique_revealed_ratio']} "
              f"top={verdict['top_revealed_token_id']}@"
              f"{verdict['top_revealed_token_share']} "
              f"median_per_row={verdict['median_distinct_tokens_per_row']}")
        if not verdict["degenerate"]:
            kept.append(str(path))
            continue
        voided.append(str(path))
        if not args.apply:
            continue
        manifest_path = manifest_for(path)
        if manifest_path.exists():
            manifest = C.load_json(manifest_path)
            manifest["artifact_status"] = "VOID"
            manifest["void_reason"] = verdict["reason"]
            manifest["void_diagnostics"] = {
                key: verdict[key] for key in (
                    "revealed_tokens", "unique_revealed_tokens",
                    "unique_revealed_ratio", "top_revealed_token_id",
                    "top_revealed_token_share", "eos_revealed_share",
                    "median_distinct_tokens_per_row",
                )
            }
            if args.note:
                manifest["void_note"] = args.note
            C.dump_json(manifest, manifest_path)
            print(f"[void]   manifest annotated -> {manifest_path.name}")
        target = Path(str(path) + VOID_SUFFIX)
        path.rename(target)
        print(f"[void]   tensor renamed -> {target.name}")

    print(f"\n[void] degenerate: {len(voided)}  usable: {len(kept)}"
          f"{'' if args.apply else '  (report only; pass --apply to record)'}")


if __name__ == "__main__":
    main()

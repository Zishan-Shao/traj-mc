"""Audit matched clean windows and distribution properties across schemes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajmc import common as C
from trajmc.sampling import uniform_grid


def parse_artifacts(values: list[str]) -> dict[str, str]:
    artifacts = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label or not path or label in artifacts:
            raise ValueError(f"invalid or duplicate artifact: {value!r}")
        artifacts[label] = str(Path(path).expanduser().resolve())
    return artifacts


def audit_blob(label: str, blob: dict, reference: torch.Tensor) -> dict:
    clean = blob["windows_pre"].long()
    states = blob["input_ids"].long()
    # A task-prompt Actual-State rollout is a control, not one of the matched
    # arms: its obligation is to resemble deployment, not to share a corpus.
    # Window identity stays binding on every arm that carries a single-variable
    # claim.
    matched_arm = not blob.get("prompt_source")
    if matched_arm and not torch.equal(clean, reference):
        raise AssertionError(f"{label}: clean windows differ from the reference arm")
    if states.shape != clean.shape:
        raise AssertionError(f"{label}: input and clean shapes differ")
    if not matched_arm:
        # Timestep grid and prefix invariants below assume the matched-arm
        # construction; a task rollout carries its own diagnostics instead.
        masks = states.eq(int(blob["mask_id"]))
        prefix_length = int(blob.get("prefix_length", 0))
        return {
            "scheme": blob.get("scheme", blob.get("arm")),
            "shape": list(states.shape),
            "clean_windows_match": False,
            "matched_arm": False,
            "prompt_source": blob.get("prompt_source"),
            "mean_full_mask_ratio": float(masks.float().mean()),
            "mean_suffix_mask_ratio": float(masks[:, prefix_length:].float().mean()),
            "prefix_length": prefix_length,
            "model_prediction_feedback": bool(blob.get("model_prediction_feedback")),
            "top_revealed_token_share": blob.get("top_revealed_token_share"),
            "eos_revealed_share": blob.get("eos_revealed_share"),
        }

    scheme = blob.get("scheme", blob.get("arm"))
    mask_id = int(blob["mask_id"])
    prefix_length = int(blob.get("prefix_length", 0))
    masks = states.eq(mask_id)
    visible = ~masks
    visible_mismatches = int((visible & states.ne(clean)).sum())
    prefix_equal = bool(torch.equal(
        states[:, :prefix_length], clean[:, :prefix_length]
    ))
    timesteps = torch.tensor(blob.get("per_sample_t", []), dtype=torch.float64)
    grid_exact = None
    if scheme in {"grid_t", "grid_t_prefix", "rollout"}:
        grid_exact = bool(torch.equal(timesteps, uniform_grid(states.shape[0])))
        if not grid_exact:
            raise AssertionError(f"{label}: timestep grid is not exact")
    if scheme != "rollout" and visible_mismatches:
        raise AssertionError(f"{label}: offline corruption changed visible tokens")
    if prefix_length and not prefix_equal:
        raise AssertionError(f"{label}: visible prefix changed")

    return {
        "scheme": scheme,
        "shape": list(states.shape),
        "clean_windows_match": bool(matched_arm),
        "matched_arm": bool(matched_arm),
        "prompt_source": blob.get("prompt_source"),
        "mean_t": float(timesteps.mean()) if timesteps.numel() else None,
        "mean_full_mask_ratio": float(masks.float().mean()),
        "mean_suffix_mask_ratio": float(
            masks[:, prefix_length:].float().mean()
        ),
        "prefix_length": prefix_length,
        "prefix_byte_identical": prefix_equal,
        "grid_exact": grid_exact,
        "visible_prediction_mismatches_vs_clean": visible_mismatches,
        "model_prediction_feedback": bool(blob.get("model_prediction_feedback")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib", action="append", required=True, help="repeat LABEL=CALIBRATION_PT"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    artifacts = parse_artifacts(args.calib)
    loaded = {
        label: torch.load(path, map_location="cpu", weights_only=False)
        for label, path in artifacts.items()
    }
    matched = [b for b in loaded.values() if not b.get("prompt_source")]
    if not matched:
        raise AssertionError("no matched arm to use as the window reference")
    first = matched[0]["windows_pre"].long()
    rows = {
        label: audit_blob(label, blob, first)
        for label, blob in loaded.items()
    }
    result = {
        "gate": "PASS",
        "all_clean_windows_byte_identical": all(
            row["clean_windows_match"] for row in rows.values()
            if row.get("matched_arm", True)
        ),
        "artifacts": artifacts,
        "schemes": rows,
    }
    if args.out:
        C.dump_json(result, args.out)
    else:
        import json

        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Layer output-reconstruction error on real generation states (RQ2).

Each compressed arm replaces ``W`` with ``A B``.  This tool replays a set of
held-out *actual* reverse-sampler states through the dense model and measures,
per compressed Linear,

    E_gen = E_h ||(W - A B) h||^2 / E_h ||W h||^2,

where ``h`` is that layer's real input at those states.  The dense forward pass
supplies ``W h`` for free as the layer's own output, so each arm costs one extra
low-rank matmul per layer rather than a second dense pass.

Bootstrap unit is the *prompt*: a rollout artifact holds exactly one state per
clean window, so rows are independent and multiple steps of one trajectory are
never treated as independent samples.  Deltas against the clean baseline use a
paired bootstrap over the shared row index.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trajmc import common as C
from trajmc.compression import _install_lm_head_stop, _stats_forward


def parse_labelled(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label or not path or label in parsed:
            raise ValueError(f"invalid or duplicate label: {value!r}")
        parsed[label] = str(Path(path).expanduser().resolve())
    return parsed


def load_factors(weights_dir: str, layers, device, dtype=torch.bfloat16):
    """Load the ``A``/``B`` pair saved by ``trajmc-compress`` for each layer."""
    factors = {}
    for name, _ in layers:
        prefix = name.replace(".", "_")
        a_path = os.path.join(weights_dir, f"{prefix}_A.pt")
        b_path = os.path.join(weights_dir, f"{prefix}_B.pt")
        if not (os.path.exists(a_path) and os.path.exists(b_path)):
            continue
        A = torch.load(a_path, map_location="cpu").to(device=device, dtype=dtype)
        B = torch.load(b_path, map_location="cpu").to(device=device, dtype=dtype)
        factors[name] = (A, B)
    if not factors:
        raise ValueError(f"no A/B factor pairs found under {weights_dir}")
    return factors


@torch.no_grad()
def measure(model, states, layers, factors, device, bins: int,
            position_start: int = 0):
    """Return per-row, per-layer squared error and dense output energy.

    ``position_start`` restricts the accumulation to sequence positions at or
    after it.  With a deployment prompt of 1112 tokens and a 256-token
    generation region, 81% of an unrestricted average is carried by clean
    prompt positions, which dilutes any difference between calibration arms.
    Both readings are worth reporting: the whole state is what the layer
    actually processes, the generation region is where the compressed model's
    output is consumed.
    """
    names = [name for name, _ in layers if name in factors]
    modules = {name: module for name, module in layers if name in factors}
    n_rows, n_layers = states.shape[0], len(names)
    numerator = np.zeros((n_rows, n_layers), dtype=np.float64)
    denominator = np.zeros((n_rows, n_layers), dtype=np.float64)
    index_of = {name: index for index, name in enumerate(names)}
    groups = [names[start::bins] for start in range(bins)]

    row_num = np.zeros(n_layers, dtype=np.float64)
    row_den = np.zeros(n_layers, dtype=np.float64)

    def make_hook(name):
        column = index_of[name]
        A, B = factors[name]
        bias = modules[name].bias

        def hook(_module, inputs, output):
            hidden = inputs[0]
            if position_start:
                hidden = hidden[:, position_start:, :]
                output = output[:, position_start:, :]
            dense = output if bias is None else output - bias
            low_rank = F.linear(F.linear(hidden, B), A)
            row_num[column] += float(
                (dense - low_rank).to(torch.float32).pow(2).sum()
            )
            row_den[column] += float(dense.to(torch.float32).pow(2).sum())

        return hook

    for group_index, group in enumerate(groups):
        if not group:
            continue
        handles = [modules[name].register_forward_hook(make_hook(name))
                   for name in group]
        stop_handle = _install_lm_head_stop(model)
        try:
            for row in range(n_rows):
                row_num[:] = 0.0
                row_den[:] = 0.0
                _stats_forward(model, states[row : row + 1].to(device), {})
                columns = [index_of[name] for name in group]
                numerator[row, columns] = row_num[columns]
                denominator[row, columns] = row_den[columns]
                if row % 32 == 0:
                    print(f"  pass {group_index + 1}/{len(groups)} "
                          f"row {row}/{n_rows}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
            if stop_handle is not None:
                stop_handle.remove()
    return names, numerator, denominator


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    total = denominator.sum()
    return float(numerator.sum() / total) if total else float("nan")


def stage_masks(mask_ratios: np.ndarray) -> dict[str, np.ndarray]:
    """Split rows into early/middle/late by remaining-MASK fraction."""
    low, high = np.quantile(mask_ratios, [1 / 3, 2 / 3])
    return {
        "late": mask_ratios <= low,        # few MASKs left: end of generation
        "middle": (mask_ratios > low) & (mask_ratios < high),
        "early": mask_ratios >= high,      # mostly MASK: start of generation
    }


def paired_bootstrap(base_num, base_den, other_num, other_den, draws, seed):
    """Paired bootstrap over rows for ``E_base - E_other``."""
    rng = np.random.default_rng(seed)
    n_rows = base_num.shape[0]
    observed = ratio(base_num, base_den) - ratio(other_num, other_den)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        rows = rng.integers(0, n_rows, n_rows)
        samples[draw] = (
            base_num[rows].sum() / base_den[rows].sum()
            - other_num[rows].sum() / other_den[rows].sum()
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    crossings = float(np.mean(samples <= 0.0)) if observed > 0 else float(
        np.mean(samples >= 0.0)
    )
    return {
        "delta_E_gen": observed,
        "ci95": [float(low), float(high)],
        "bootstrap_sign_crossing_fraction": crossings,
        "improves": bool(observed > 0 and low > 0),
        "draws": draws,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(C.BACKENDS), required=True)
    parser.add_argument("--states", required=True,
                        help="held-out rollout calibration .pt (real sampler states)")
    parser.add_argument("--weights", action="append", required=True,
                        help="repeat LABEL=WEIGHTS_DIR")
    parser.add_argument("--baseline", default="clean",
                        help="label used as the reference arm for deltas")
    parser.add_argument("--oracle", default="actual",
                        help="label used as the upper bound in GapClosed")
    parser.add_argument("--calib_manifest", action="append", default=[],
                        help="calibration manifest .json whose prompts must not "
                             "overlap the evaluation states; repeatable")
    parser.add_argument("--allow_prompt_overlap", action="store_true")
    parser.add_argument("--allow_offline_states", action="store_true",
                        help="permit a non-rollout states artifact")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--layer_type", choices=["all", "attn", "mlp"], default="all")
    parser.add_argument("--max_states", type=int, default=0)
    parser.add_argument("--bins", type=int, default=1,
                        help="split layers over N forward passes to bound the "
                             "GPU memory held by A/B factors")
    parser.add_argument("--positions", choices=["all", "generation"],
                        default="all",
                        help="accumulate over every position, or only the "
                             "generation region after the prompt")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    arms = parse_labelled(args.weights)
    if args.baseline not in arms:
        parser.error(f"--baseline {args.baseline!r} is not one of {sorted(arms)}")
    if args.bins < 1:
        parser.error("--bins must be positive")

    backend = C.get_backend(args.backend)
    blob = torch.load(args.states, map_location="cpu", weights_only=False)
    if blob.get("backend") not in (None, backend.name):
        raise ValueError(f"states artifact belongs to backend {blob.get('backend')}")
    if blob.get("scheme") != "rollout" and not args.allow_offline_states:
        raise ValueError(
            f"states artifact scheme is {blob.get('scheme')!r}; E_gen must be "
            "measured on real rollout states (pass --allow_offline_states to override)"
        )
    states = blob["input_ids"]
    if args.max_states:
        states = states[: args.max_states]
    state_hashes = list(blob.get("window_hashes_prenoise") or [])[: states.shape[0]]

    overlaps = {}
    for path in args.calib_manifest:
        manifest = C.load_json(path)
        shared = set(state_hashes) & set(manifest.get("window_hashes_prenoise") or [])
        overlaps[os.path.basename(path)] = len(shared)
        if shared and not args.allow_prompt_overlap:
            raise ValueError(
                f"{len(shared)} evaluation prompts also appear in {path}; "
                "E_gen must use held-out prompts (--allow_prompt_overlap to override)"
            )

    prefix_length = int(blob.get("prefix_length") or 0)
    suffix = states[:, prefix_length:]
    mask_ratios = suffix.eq(backend.mask_id).float().mean(dim=1).numpy()
    stages = stage_masks(mask_ratios)

    model, _ = C.load_model(
        backend,
        model_path=args.model_path,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        device=args.device,
    )
    layers = list(C.iter_target_linears(model, backend, args.layer_type))
    position_start = prefix_length if args.positions == "generation" else 0
    print(f"[recon] states={tuple(states.shape)} prefix={prefix_length} "
          f"positions={args.positions} (from index {position_start}) "
          f"target_linears={len(layers)} arms={sorted(arms)}", flush=True)

    results = {}
    raw = {}
    layer_names = None
    for label, weights_dir in arms.items():
        summary_path = os.path.join(weights_dir, "compression_summary.json")
        summary = C.load_json(summary_path) if os.path.exists(summary_path) else {}
        if summary.get("backend") not in (None, backend.name):
            raise ValueError(f"{label} was compressed for {summary.get('backend')}")
        print(f"[recon] arm {label}: {weights_dir}", flush=True)
        factors = load_factors(weights_dir, layers, args.device)
        names, numerator, denominator = measure(
            model, states, layers, factors, args.device, args.bins,
            position_start=position_start,
        )
        if layer_names is None:
            layer_names = names
        elif names != layer_names:
            raise ValueError(
                f"arm {label} compressed a different layer set than "
                f"{next(iter(results))}; the comparison would not be matched"
            )
        raw[label] = (numerator, denominator)
        per_layer = {
            name: ratio(numerator[:, column], denominator[:, column])
            for column, name in enumerate(names)
        }
        results[label] = {
            "weights": weights_dir,
            "calib_scheme": summary.get("calib_scheme", summary.get("arm")),
            "ratio": summary.get("ratio"),
            "kept_fraction": summary.get("kept_fraction"),
            "decomp": summary.get("decomp"),
            "n_layers_measured": len(names),
            "E_gen": ratio(numerator, denominator),
            "E_gen_mean_over_layers": float(np.mean(list(per_layer.values()))),
            "per_layer": per_layer,
            "per_stage": {
                stage: ratio(numerator[rows], denominator[rows])
                for stage, rows in stages.items()
            },
        }
        del factors
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[recon] {label}: E_gen={results[label]['E_gen']:.6f}", flush=True)

    baseline_num, baseline_den = raw[args.baseline]
    baseline_error = results[args.baseline]["E_gen"]
    bootstrap = {}
    for label in arms:
        if label == args.baseline:
            continue
        other_num, other_den = raw[label]
        bootstrap[label] = paired_bootstrap(
            baseline_num, baseline_den, other_num, other_den,
            args.bootstrap, args.bootstrap_seed,
        )

    gap_closed = {}
    if args.oracle in results and args.oracle != args.baseline:
        span = baseline_error - results[args.oracle]["E_gen"]
        for label in arms:
            if label in (args.baseline, args.oracle):
                continue
            gap_closed[label] = (
                (baseline_error - results[label]["E_gen"]) / span
                if span else None
            )

    rows_path = Path(args.out).with_suffix(".rows.npz")
    np.savez_compressed(
        rows_path,
        layers=np.array(layer_names),
        mask_ratios=mask_ratios,
        **{f"{label}_num": raw[label][0] for label in arms},
        **{f"{label}_den": raw[label][1] for label in arms},
    )

    output = {
        "definition": (
            "E_gen = sum ||(W - AB)h||^2 / sum ||Wh||^2 over held-out real "
            "reverse-sampler states; bootstrap unit is the prompt"
        ),
        "backend": backend.name,
        "model_path": args.model_path or backend.model_id,
        "layer_type": args.layer_type,
        "positions": args.positions,
        "position_start": position_start,
        "states": {
            "path": str(Path(args.states).resolve()),
            "scheme": blob.get("scheme"),
            "n_prompts": int(states.shape[0]),
            "seqlen": int(states.shape[1]),
            "prefix_length": prefix_length,
            "window_seed": blob.get("window_seed", blob.get("seed")),
            "sampling_seed": blob.get("sampling_seed"),
            "rollout_steps": blob.get("rollout_steps"),
            "sampler": blob.get("sampler"),
            "mean_suffix_mask_ratio": float(mask_ratios.mean()),
            "prompt_overlap_with_calibration": overlaps,
        },
        "stage_counts": {stage: int(rows.sum()) for stage, rows in stages.items()},
        "baseline": args.baseline,
        "oracle": args.oracle if args.oracle in results else None,
        "arms": results,
        "paired_bootstrap_vs_baseline": bootstrap,
        "gap_closed": gap_closed,
        "row_level_data": str(rows_path),
        "interpretation": (
            "RQ2 passes when E_traj < E_clean with a bootstrap CI excluding zero. "
            "GapClosed reports how much of the clean-to-actual span the arm covers."
        ),
    }
    C.dump_json(output, args.out)
    print(f"[recon] report -> {Path(args.out).resolve()}")
    for label, entry in sorted(results.items(), key=lambda kv: kv[1]["E_gen"]):
        line = f"[recon] {label}: E_gen={entry['E_gen']:.6f}"
        if label in gap_closed and gap_closed[label] is not None:
            line += f"  GapClosed={gap_closed[label]:.1%}"
        if label in bootstrap:
            ci = bootstrap[label]["ci95"]
            line += (f"  dE={bootstrap[label]['delta_E_gen']:+.6f} "
                     f"[{ci[0]:+.6f}, {ci[1]:+.6f}]")
        print(line)


if __name__ == "__main__":
    main()

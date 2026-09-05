"""Does the calibration distribution change *which* low-rank subspace is kept?

RQ1 of the Traj-SVD study.  Activation-aware SVD keeps, per Linear, a rank-``k``
input subspace determined entirely by the calibration second moment ``Sigma``.
This diagnostic recomputes that subspace from several calibration artifacts and
measures how far each one lands from the reference (normally the real-rollout
``Sigma``), so a scale-only shift cannot be mistaken for a subspace shift.

Two subspaces are reported per layer:

``kept_input``
    row space of ``B`` in ``W ~= A B``, i.e. the directions the deployed
    compressed layer actually keeps.  This is the quantity tied to the
    reconstruction error measured by :mod:`analysis.gen_reconstruction`.
``sigma_eig``
    top-``k`` eigenspace of ``Sigma`` itself.  Cheaper to interpret, but it
    ignores ``W``; reported alongside because the plan states ``D_sub`` in these
    terms.

Distances use the projector Frobenius norm, evaluated without materialising
either projector::

    ||P1 - P2||_F = sqrt(k1 + k2 - 2 ||Q1^T Q2||_F^2),

normalised by ``sqrt(k1 + k2)`` so 0 means identical and 1 means orthogonal.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from trajmc import common as C
from trajmc.compression import collect_xtx_gpu, whiten_truncate
from analysis.covariance_estimation import parse_calibrations


def select_layers(
    model,
    backend,
    suffixes: list[str],
    block_stride: int,
    explicit: list[str],
) -> list[tuple[str, nn.Linear]]:
    """Pick the Linears the curve is drawn over, in model order."""
    targets = list(C.iter_target_linears(model, backend))
    if explicit:
        by_name = dict(targets)
        missing = sorted(set(explicit) - set(by_name))
        if missing:
            raise ValueError(f"unknown target layer(s): {missing}")
        return [(name, by_name[name]) for name in explicit]
    chosen = [(n, m) for n, m in targets if n.endswith(tuple(suffixes))]
    if not chosen:
        raise ValueError(f"no target Linear ends with any of {suffixes}")
    return chosen[::block_stride]


def block_index(name: str, marker: str) -> int | None:
    """Depth of a target Linear, for the x axis of Figure 1."""
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    head = tail.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def orthonormal_basis(matrix: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (columns) for the column space of ``matrix``."""
    Q, _ = torch.linalg.qr(matrix.to(torch.float64), mode="reduced")
    return Q


def subspace_metrics(Q1: torch.Tensor, Q2: torch.Tensor) -> dict:
    """Projector distance and principal angles between two column spaces."""
    k1, k2 = Q1.shape[1], Q2.shape[1]
    overlap = Q1.transpose(0, 1) @ Q2
    squared = float(k1 + k2 - 2.0 * float(overlap.pow(2).sum()))
    distance = math.sqrt(max(squared, 0.0))
    cosines = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cosines))
    return {
        "projector_frobenius": distance,
        "normalised_projector_frobenius": distance / math.sqrt(k1 + k2),
        "mean_principal_angle_deg": float(angles.mean()),
        "median_principal_angle_deg": float(angles.median()),
        "max_principal_angle_deg": float(angles.max()),
        "captured_energy_fraction": float(overlap.pow(2).sum()) / max(k1, 1),
        "k": int(k1),
        "k_reference": int(k2),
    }


def top_eigenspace(sigma: torch.Tensor, k: int) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sigma.to(torch.float64))
    order = torch.argsort(values, descending=True)[:k]
    return vectors[:, order]


def kept_input_space(W: torch.Tensor, sigma: torch.Tensor, ratio: float,
                     decomp: str) -> torch.Tensor:
    """Orthonormal basis of the input directions the compressed layer keeps."""
    _, B, _ = whiten_truncate(W, sigma, ratio, decomp=decomp)
    return orthonormal_basis(B.transpose(0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(C.BACKENDS), required=True)
    parser.add_argument("--calib", action="append", required=True,
                        help="repeat LABEL=CALIBRATION_PT")
    parser.add_argument("--reference", required=True,
                        help="label treated as the deployment truth (normally rollout)")
    parser.add_argument("--ratio", type=float, required=True,
                        help="retained parameter fraction; fixes k per layer")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--layer", action="append", default=[],
                        help="explicit target Linear name; repeatable")
    parser.add_argument("--suffix", action="append", default=[],
                        help="layer suffix to sweep over depth (default: first "
                             "attention suffix of the backend)")
    parser.add_argument("--block_stride", type=int, default=1)
    parser.add_argument("--decomp", choices=["cholesky", "eigh"], default="cholesky")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--linalg_device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    calibrations = parse_calibrations(args.calib)
    if args.reference not in calibrations:
        parser.error("--reference must name one of the --calib labels")
    if len(calibrations) < 2:
        parser.error("supply at least two --calib artifacts to compare")
    if args.block_stride < 1:
        parser.error("--block_stride must be positive")

    backend = C.get_backend(args.backend)
    suffixes = args.suffix or [backend.attention_suffixes[0]]
    model, _ = C.load_model(
        backend,
        model_path=args.model_path,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        device=args.device,
    )
    layers = select_layers(model, backend, suffixes, args.block_stride, args.layer)
    modules = {name: module for name, module in layers}
    footprint = sum(m.in_features ** 2 * 4 for m in modules.values()) / 1024 ** 3
    print(f"[subspace] {len(layers)} layers, {footprint:.1f}GB XtX per artifact, "
          f"{footprint * len(calibrations):.1f}GB held on CPU", flush=True)

    moments = {}
    metadata = {}
    for label, path in calibrations.items():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("backend") not in (None, backend.name):
            raise ValueError(f"{label} belongs to backend {blob.get('backend')}")
        input_ids = blob["input_ids"]
        if args.max_samples:
            input_ids = input_ids[: args.max_samples]
        print(f"[subspace] XtX for {label}: {tuple(input_ids.shape)}", flush=True)
        moments[label] = collect_xtx_gpu(
            model, input_ids, modules, args.device,
            log_prefix=f"{label} ", batch_size=args.batch_size,
        )
        metadata[label] = {
            "path": path,
            "scheme": blob.get("scheme", blob.get("arm")),
            "samples": int(input_ids.shape[0]),
            "window_seed": blob.get("window_seed", blob.get("seed")),
            "sampling_seed": blob.get("sampling_seed"),
            "prefix_length": blob.get("prefix_length"),
        }
        del blob

    linalg_device = args.linalg_device if args.device.startswith("cuda") else "cpu"
    others = [label for label in calibrations if label != args.reference]
    per_layer = {label: {} for label in others}
    curve = []

    for name, module in layers:
        W = module.weight.data.to(device=linalg_device, dtype=torch.float32)
        out_dim, in_dim = W.shape
        k = min(C.rank_from_ratio(args.ratio, out_dim, in_dim), out_dim, in_dim)
        bases = {}
        for label in calibrations:
            sigma = moments[label][name].to(linalg_device)
            bases[label] = {
                "kept_input": kept_input_space(W, sigma, args.ratio, args.decomp),
                "sigma_eig": top_eigenspace(sigma, k),
            }
            del sigma
        reference_sigma = moments[args.reference][name].to(torch.float64)
        reference_norm = float(torch.linalg.matrix_norm(reference_sigma, ord="fro"))
        for label in others:
            delta = moments[label][name].to(torch.float64) - reference_sigma
            row = {
                "block_index": block_index(name, backend.block_marker),
                "in_features": int(in_dim),
                "out_features": int(out_dim),
                "rank_k": int(k),
                "sigma_relative_frobenius": (
                    float(torch.linalg.matrix_norm(delta, ord="fro")) / reference_norm
                ),
                "sigma_relative_trace": abs(
                    float(torch.trace(moments[label][name].to(torch.float64)))
                    / float(torch.trace(reference_sigma)) - 1.0
                ),
            }
            for space in ("kept_input", "sigma_eig"):
                metrics = subspace_metrics(
                    bases[label][space], bases[args.reference][space]
                )
                row[space] = metrics
            per_layer[label][name] = row
            curve.append({
                "layer": name,
                "block_index": row["block_index"],
                "label": label,
                "kept_input_distance": row["kept_input"][
                    "normalised_projector_frobenius"
                ],
                "sigma_eig_distance": row["sigma_eig"][
                    "normalised_projector_frobenius"
                ],
                "sigma_relative_frobenius": row["sigma_relative_frobenius"],
            })
            del delta
        del W, bases, reference_sigma
        if linalg_device == "cuda":
            torch.cuda.empty_cache()
        print(f"[subspace] {name} done", flush=True)

    def mean_of(label, space, key):
        rows = per_layer[label].values()
        return sum(row[space][key] for row in rows) / len(rows)

    comparisons = {
        label: {
            "layers": per_layer[label],
            "aggregate": {
                "mean_kept_input_normalised_distance": mean_of(
                    label, "kept_input", "normalised_projector_frobenius"),
                "mean_kept_input_principal_angle_deg": mean_of(
                    label, "kept_input", "mean_principal_angle_deg"),
                "mean_sigma_eig_normalised_distance": mean_of(
                    label, "sigma_eig", "normalised_projector_frobenius"),
                "mean_sigma_relative_frobenius": sum(
                    row["sigma_relative_frobenius"]
                    for row in per_layer[label].values()
                ) / len(per_layer[label]),
            },
        }
        for label in others
    }
    ranking = sorted(
        comparisons,
        key=lambda label: comparisons[label]["aggregate"][
            "mean_kept_input_normalised_distance"
        ],
    )

    output = {
        "definition": (
            "distance from each calibration's rank-k kept input subspace to the "
            "reference calibration's, per compressed Linear"
        ),
        "backend": backend.name,
        "model_path": args.model_path or backend.model_id,
        "ratio": args.ratio,
        "decomp": args.decomp,
        "reference": args.reference,
        "layer_suffixes": suffixes,
        "block_stride": args.block_stride,
        "layers": [name for name, _ in layers],
        "calibrations": metadata,
        "comparisons": comparisons,
        "closest_to_reference_first": ranking,
        "per_layer_curve": curve,
        "interpretation": (
            "RQ1 passes when the Traj arm's mean kept-input distance to the "
            "rollout reference is below the clean arm's, and the gap is not "
            "explained by sigma_relative_trace alone (a pure scale shift)."
        ),
    }
    C.dump_json(output, args.out)
    print(f"[subspace] report -> {Path(args.out).resolve()}")
    for label in ranking:
        aggregate = comparisons[label]["aggregate"]
        print(f"[subspace] {label}: kept-input d="
              f"{aggregate['mean_kept_input_normalised_distance']:.4f}  "
              f"angle={aggregate['mean_kept_input_principal_angle_deg']:.1f}deg  "
              f"sigma dF={aggregate['mean_sigma_relative_frobenius']:.4f}")


if __name__ == "__main__":
    main()

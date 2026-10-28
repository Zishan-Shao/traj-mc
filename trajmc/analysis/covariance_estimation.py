"""Compare calibration covariance estimators on matched dense-model activations.

For each selected Linear, this diagnostic accumulates the exact second moment
on a deterministic coordinate subspace of its input activations.  A larger,
independent calibration artifact should be supplied as the reference.  Running
multiple random-t seeds makes estimator bias/variance directly measurable.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from trajmc import common as C


class _StopBeforeHead(RuntimeError):
    pass


def parse_calibrations(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label or not path or label in parsed:
            raise ValueError(f"invalid or duplicate calibration label: {value!r}")
        parsed[label] = str(Path(path).expanduser().resolve())
    return parsed


def choose_layers(model, backend, requested: list[str]) -> list[tuple[str, nn.Linear]]:
    targets = list(C.iter_target_linears(model, backend))
    if requested:
        by_name = dict(targets)
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise ValueError(f"unknown target layer(s): {missing}")
        return [(name, by_name[name]) for name in requested]
    indices = sorted({0, len(targets) // 2, len(targets) - 1})
    return [targets[index] for index in indices]


def coordinate_subspaces(
    layers: list[tuple[str, nn.Linear]], sketch_dim: int, seed: int
) -> dict[str, torch.Tensor]:
    coordinates = {}
    for name, layer in layers:
        digest = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed + digest)
        dimension = min(sketch_dim, layer.in_features)
        coordinates[name] = torch.randperm(
            layer.in_features, generator=generator
        )[:dimension]
    return coordinates


@torch.no_grad()
def collect_moments(
    model,
    input_ids: torch.Tensor,
    layers: list[tuple[str, nn.Linear]],
    coordinates: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    sums = {}
    counts = {name: 0 for name, _ in layers}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            index = coordinates[name].to(x.device)
            projected = x.index_select(-1, index).reshape(-1, index.numel()).float()
            if name not in sums:
                sums[name] = torch.zeros(
                    index.numel(), index.numel(), device=x.device, dtype=torch.float64
                )
            sums[name].addmm_(projected.T.double(), projected.double())
            counts[name] += projected.shape[0]

        return hook

    for name, layer in layers:
        handles.append(layer.register_forward_hook(make_hook(name)))

    head = C.get_output_head(model)

    def stop(_module, _inputs):
        raise _StopBeforeHead()

    stop_handle = (
        head.register_forward_pre_hook(stop) if isinstance(head, nn.Linear) else None
    )
    try:
        for start in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[start : start + batch_size].to(device)
            try:
                model(batch)
            except _StopBeforeHead:
                pass
    finally:
        for handle in handles:
            handle.remove()
        if stop_handle is not None:
            stop_handle.remove()

    moments = {
        name: (sums[name] / counts[name]).cpu()
        for name, _ in layers
    }
    return moments, counts


def compare_moments(estimate, reference):
    rows = {}
    for name in sorted(reference):
        ref = reference[name]
        delta = estimate[name] - ref
        ref_fro = torch.linalg.matrix_norm(ref, ord="fro")
        ref_spectral = torch.linalg.matrix_norm(ref, ord=2)
        rows[name] = {
            "relative_frobenius_error": float(
                torch.linalg.matrix_norm(delta, ord="fro") / ref_fro
            ),
            "relative_spectral_error": float(
                torch.linalg.matrix_norm(delta, ord=2) / ref_spectral
            ),
            "relative_trace_error": float(
                abs(torch.trace(estimate[name]) / torch.trace(ref) - 1.0)
            ),
        }
    metric_names = next(iter(rows.values()))
    aggregate = {
        f"mean_{metric}": sum(row[metric] for row in rows.values()) / len(rows)
        for metric in metric_names
    }
    return rows, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(C.BACKENDS), required=True)
    parser.add_argument(
        "--calib", action="append", required=True, help="repeat LABEL=CALIBRATION_PT"
    )
    parser.add_argument("--reference", required=True, help="label used as covariance truth")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--layer", action="append", default=[])
    parser.add_argument("--sketch_dim", type=int, default=256)
    parser.add_argument("--sketch_seed", type=int, default=2027)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    calibrations = parse_calibrations(args.calib)
    if args.reference not in calibrations:
        parser.error("--reference must name one of the --calib labels")
    if args.sketch_dim <= 0 or args.batch_size <= 0:
        parser.error("--sketch_dim and --batch_size must be positive")

    backend = C.get_backend(args.backend)
    model, _ = C.load_model(
        backend,
        model_path=args.model_path,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        device=args.device,
    )
    layers = choose_layers(model, backend, args.layer)
    coordinates = coordinate_subspaces(layers, args.sketch_dim, args.sketch_seed)

    all_moments = {}
    metadata = {}
    for label, path in calibrations.items():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("backend") not in (None, backend.name):
            raise ValueError(f"{label} belongs to backend {blob.get('backend')}")
        input_ids = blob["input_ids"]
        if args.max_samples:
            input_ids = input_ids[: args.max_samples]
        print(f"[covariance] {label}: {tuple(input_ids.shape)}", flush=True)
        moments, counts = collect_moments(
            model, input_ids, layers, coordinates, args.device, args.batch_size
        )
        all_moments[label] = moments
        metadata[label] = {
            "path": path,
            "scheme": blob.get("scheme", blob.get("arm")),
            "samples": input_ids.shape[0],
            "activation_rows_per_layer": counts,
        }

    reference = all_moments[args.reference]
    comparisons = {}
    for label, moments in all_moments.items():
        layer_rows, aggregate = compare_moments(moments, reference)
        comparisons[label] = {"layers": layer_rows, "aggregate": aggregate}

    output = {
        "definition": "dense-model input-activation second-moment estimation",
        "reference": args.reference,
        "backend": backend.name,
        "model_path": args.model_path or backend.model_id,
        "coordinate_subspace_dimension": args.sketch_dim,
        "coordinate_seed": args.sketch_seed,
        "layers": [name for name, _ in layers],
        "calibrations": metadata,
        "comparisons": comparisons,
        "interpretation": (
            "Lower error is better. Repeat iid random-t with independent seeds; "
            "use a larger independent calibration artifact as reference."
        ),
    }
    C.dump_json(output, args.out)
    print(f"[covariance] report -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

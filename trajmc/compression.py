"""Shared activation-aware low-rank compression for LLaDA and Dream.

The arm identity lives entirely in the calib file (built by calib/build_calib.py).
This module is arm-agnostic: same whitening math, same truncation, same pipeline.

Faithful port of official SVD-LLM whitening (see docs/svdllm_port_diff.md):
  - single forward + forward hooks accumulate X^T X incrementally (O(d^2)/layer),
    no inps/outs buffering;
  - decomposition: torch.linalg.cholesky in fp64 with the official except branch
    (+(-lambda_min+1e-6) I on failure); eigh is a verified-equivalent backup,
    DEFAULT OFF (--decomp cholesky);
  - rank k = int(ratio * out * in / (out + in)); full SVD after whitening;
  - compresses only backend-declared block-internal Linears; lm_head stays dense.

Outputs (results/compress/ and the weights dir): per-layer A/B .pt + summary.json.
"""
import os
import time
import argparse
import hashlib
import torch
import torch.nn as nn

from . import common as C


class _StopBeforeLMHead(RuntimeError):
    pass


def _install_lm_head_stop(model):
    """Skip the enormous vocab projection when only block activations are needed."""
    head = C.get_output_head(model)
    if not isinstance(head, nn.Linear):
        return None

    def stop(_module, _inputs):
        raise _StopBeforeLMHead()

    return head.register_forward_pre_hook(stop)


def _stats_forward(model, input_ids, kwargs):
    try:
        model(input_ids, **kwargs)
    except _StopBeforeLMHead:
        return


# ── whitened truncation (official math). Reused by the equivalence self-check. ──
def cholesky_factor(XtX, solve_dtype=torch.float64):
    """
    Official Cholesky with except branch. Returns L (solve_dtype) s.t. L L^T = XtX.
    """
    raw = XtX.to(solve_dtype)
    try:
        L = torch.linalg.cholesky(raw)
    except Exception:
        # official: shift by (-lambda_min + 1e-6) I then retry
        ev = torch.linalg.eigvalsh(raw)
        raw = raw + (-ev[0] + 1e-6) * torch.eye(raw.shape[0], dtype=solve_dtype,
                                                device=raw.device)
        L = torch.linalg.cholesky(raw)
    return L


def eigh_factor(XtX, solve_dtype=torch.float64):
    """
    Backup symmetric square root via eigh: L = V diag(sqrt(lambda)) s.t. L L^T = XtX.
    Different matrix than the Cholesky factor, but yields the SAME optimal rank-k W'
    because the truncation objective trace((W-W') XtX (W-W')^T) depends only on XtX.
    """
    raw = XtX.to(solve_dtype)
    ev, V = torch.linalg.eigh(raw)
    ev = ev.clamp(min=0)
    L = V * ev.sqrt().unsqueeze(0)
    return L


def whiten_truncate(W, XtX, ratio, decomp="cholesky", solve_dtype=torch.float64,
                    out_dtype=torch.float32):
    """
    Returns (A, B, k) with A@B the rank-k activation-aware approximation of W.
    Precision ladder matches official: factor/inv in solve_dtype (fp64),
    whitened SVD in out_dtype (fp32).
    """
    out_dim, in_dim = W.shape
    if decomp == "identity":
        # Plain weight SVD: the "no activation statistics" control.  Equivalent
        # to whitening with XtX = I, but skips the factor and the solve.
        Wf = W.to(out_dtype)
        U, S, Vt = torch.linalg.svd(Wf, full_matrices=False)
        k = min(C.rank_from_ratio(ratio, out_dim, in_dim), S.shape[0])
        sqrt_s = S[:k].sqrt()
        A = (U[:, :k] * sqrt_s).contiguous()
        B = (torch.diag(sqrt_s) @ Vt[:k, :]).contiguous()
        return A, B, k
    if decomp == "cholesky":
        L = cholesky_factor(XtX, solve_dtype)
    elif decomp == "eigh":
        L = eigh_factor(XtX, solve_dtype)
    else:
        raise ValueError(decomp)

    Lf = L.to(out_dtype)
    Wf = W.to(out_dtype)

    W_scale = Wf @ Lf
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    k = C.rank_from_ratio(ratio, out_dim, in_dim)
    k = min(k, S.shape[0])

    truc_s = S[:k]
    truc_u = U[:, :k]
    # Vt_k @ inv(L), evaluated as a triangular solve. This is algebraically
    # identical to the official explicit inverse and avoids materializing a
    # second dxd matrix (important for 12288-wide MLP projections).
    try:
        if decomp == "cholesky":
            solved = torch.linalg.solve_triangular(
                Lf.transpose(0, 1), Vt[:k, :].transpose(0, 1), upper=True
            )
        else:
            solved = torch.linalg.solve(
                Lf.transpose(0, 1), Vt[:k, :].transpose(0, 1)
            )
        truc_v = solved.transpose(0, 1)
    except Exception:
        Lf = Lf + 1e-6 * torch.eye(Lf.shape[0], dtype=Lf.dtype, device=Lf.device)
        if decomp == "cholesky":
            solved = torch.linalg.solve_triangular(
                Lf.transpose(0, 1), Vt[:k, :].transpose(0, 1), upper=True
            )
        else:
            solved = torch.linalg.solve(
                Lf.transpose(0, 1), Vt[:k, :].transpose(0, 1)
            )
        truc_v = solved.transpose(0, 1)
    sqrt_s = truc_s.sqrt()
    A = (truc_u * sqrt_s).contiguous()               # out x k
    B = (torch.diag(sqrt_s) @ truc_v).contiguous()   # k x in
    return A, B, k


def collect_xtx_gpu(model, input_ids, target_mods, device, log_prefix="",
                    batch_size=1):
    """
    Accumulate X^T X for {name: module} in ONE forward pass, ON GPU (no per-forward
    D2H copy), offloading to CPU only at the end. Caller must ensure this group's
    total XtX (sum of in_features^2 * 4 bytes) fits on the GPU beside the model.
    Returns {name: cov_cpu_fp32 (already divided by token count)}.
    """
    cov, token_count = {}, {}
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    def make_hook(name, in_dim):
        def hook(module, inp, out):
            x = inp[0].detach().reshape(-1, in_dim).float()
            if name not in cov:
                cov[name] = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
                token_count[name] = 0
            cov[name].addmm_(x.t(), x)
            token_count[name] += x.shape[0]
        return hook

    handles = [m.register_forward_hook(make_hook(n, m.in_features))
               for n, m in target_mods.items()]
    stop_handle = _install_lm_head_stop(model)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            end = min(start + batch_size, input_ids.shape[0])
            _stats_forward(model, input_ids[start:end].to(device), {})
            if start % 200 == 0:
                print(f"  {log_prefix}collect {start}/{input_ids.shape[0]} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    if stop_handle is not None:
        stop_handle.remove()
    out = {}
    for n in list(cov):
        out[n] = (cov[n] / max(token_count[n], 1)).cpu()
        cov[n] = None
    del cov
    torch.cuda.empty_cache()
    return out


def plan_bins(targets, budget_bytes):
    """Greedy-bin target linears so each bin's XtX total <= budget (fits on GPU)."""
    bins, cur, cur_b = [], {}, 0
    for name, mod in targets:
        b = mod.in_features * mod.in_features * 4
        if cur and cur_b + b > budget_bytes:
            bins.append(cur)
            cur, cur_b = {}, 0
        cur[name] = mod
        cur_b += b
    if cur:
        bins.append(cur)
    return bins


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fingerprint_weight_dir(path):
    """Hash ordered factor file names and bytes; excludes mutable summaries."""
    h = hashlib.sha256()
    files = sorted(
        fn for fn in os.listdir(path) if fn.endswith(("_A.pt", "_B.pt"))
    )
    for fn in files:
        h.update(fn.encode("utf-8"))
        with open(os.path.join(path, fn), "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest(), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(C.BACKENDS), required=True)
    ap.add_argument("--calib", default=None,
                    help="calibration .pt from trajmc-calibrate; "
                         "not used (and not required) by --decomp identity")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--layer_type", choices=["all", "attn", "mlp"], default="all")
    ap.add_argument("--decomp", choices=["cholesky", "eigh", "identity"],
                    default="cholesky",
                    help="identity = plain weight SVD, no calibration activations")
    ap.add_argument("--model_path", type=str, default=None,
                    help="local checkpoint or Hugging Face id")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--nsamples", type=int, default=0,
                    help="cap calibration to the first N windows (0 = use all)")
    ap.add_argument("--xtx_budget_gb", type=float, default=0.0,
                    help="max GB of XtX per pass; 0 chooses from free GPU memory")
    ap.add_argument("--xtx_headroom_gb", type=float, default=6.0,
                    help="GPU memory reserved for activations in automatic mode")
    ap.add_argument("--linalg_device", choices=["cpu", "cuda"], default="cpu",
                    help="device for Cholesky/SVD; cuda is intended for H100/H200 smoke runs")
    ap.add_argument("--offload_model_before_linalg", action="store_true",
                    help="after statistic collection, move the dense model to CPU "
                         "to leave GPU memory for Cholesky/SVD")
    ap.add_argument("--save_dtype", choices=["float32", "bfloat16"], default="float32",
                    help="factor-file dtype; evaluation casts factors to bfloat16, so "
                         "bfloat16 stores the deployed values without changing runtime math")
    ap.add_argument("--run_id", default=None,
                    help="immutable artifact id; required for collision-free smoke runs")
    args = ap.parse_args()

    backend = C.get_backend(args.backend)
    weight_only = args.decomp == "identity"
    if weight_only and args.calib:
        ap.error("--decomp identity ignores activations; drop --calib")
    if not weight_only and not args.calib:
        ap.error("--calib is required unless --decomp identity")
    model_path = args.model_path or backend.model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ghash = C.git_hash()
    os.makedirs(args.save_path, exist_ok=True)
    t0 = time.time()

    # ── load calib (carries the arm identity) ─────────────────────────────────
    if weight_only:
        blob = {}
        input_ids = None
        arm = "weight_svd"
        print(f"[compress] backend={backend.name} arm={arm} calib=<none> "
              f"ratio={args.ratio} layer_type={args.layer_type} "
              f"decomp={args.decomp}")
    else:
        blob = torch.load(args.calib, map_location="cpu")
        calib_backend = blob.get("backend")
        if calib_backend and calib_backend != backend.name:
            raise ValueError(
                f"calibration backend={calib_backend!r} does not match "
                f"--backend={backend.name!r}"
            )
        input_ids = blob["input_ids"]
        if args.nsamples and args.nsamples < input_ids.shape[0]:
            input_ids = input_ids[: args.nsamples]
        arm = blob.get("arm", "unknown")
        print(f"[compress] backend={backend.name} arm={arm} "
              f"calib={os.path.basename(args.calib)} "
              f"N={input_ids.shape[0]} ratio={args.ratio} "
              f"layer_type={args.layer_type} decomp={args.decomp}")

    # ── load model ────────────────────────────────────────────────────────────
    model, _tokenizer = C.load_model(backend, model_path, device=device)
    n_targets = C.count_target_linears(model, backend, args.layer_type)
    if args.layer_type == "all" and n_targets != backend.expected_linears:
        raise RuntimeError(
            f"{backend.name} target graph changed: found {n_targets}, "
            f"expected {backend.expected_linears}"
        )
    print(f"[compress] target linears ({args.layer_type}) = {n_targets}")

    # ── X^T X collection: GPU-resident accumulation, memory-binned ────────────
    # XtX stays on GPU (no per-forward D2H copy) and is offloaded once per bin.
    # all-linear XtX (~32GB) exceeds a 46GB GPU beside the 16GB model, so targets
    # are split into bins that each fit; one forward pass per bin.
    targets = list(C.iter_target_linears(model, backend, args.layer_type))
    cov = {}
    if weight_only:
        print("[compress] --decomp identity: no activation pass")
    else:
        t_collect = time.time()
        budget_gb = args.xtx_budget_gb
        if budget_gb <= 0:
            if device == "cuda":
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                budget_gb = max(
                    2.0, free_bytes / 1024 ** 3 - args.xtx_headroom_gb
                )
                print(
                    f"[compress] automatic XtX budget: "
                    f"free={free_bytes / 1024**3:.1f}GB "
                    f"total={total_bytes / 1024**3:.1f}GB "
                    f"budget={budget_gb:.1f}GB"
                )
            else:
                budget_gb = 8.0
        bins = plan_bins(targets, budget_gb * 1024 ** 3)
        print(f"[compress] {len(targets)} target linears -> {len(bins)} XtX pass(es) "
              f"(budget {budget_gb:.1f}GB/pass)")
        for bi, modules in enumerate(bins):
            gb = sum(m.in_features ** 2 * 4 for m in modules.values()) / 1024 ** 3
            print(f"[compress] XtX pass {bi+1}/{len(bins)}: "
                  f"{len(modules)} layers, {gb:.1f}GB", flush=True)
            cov.update(collect_xtx_gpu(
                model, input_ids, modules, device, log_prefix=f"p{bi+1} ",
                batch_size=args.batch_size,
            ))
        print(f"[compress] XtX collected for {len(cov)} layers "
              f"({time.time()-t_collect:.0f}s)")

    # ── whiten + truncate + replace ───────────────────────────────────────────
    if args.linalg_device == "cuda" and args.offload_model_before_linalg:
        model.to("cpu")
        torch.cuda.empty_cache()
        print("[compress] dense model offloaded to CPU before CUDA linalg", flush=True)
    t_dec = time.time()
    summary_layers = {}
    orig_params = comp_params = n_comp = n_skip = 0

    for name, mod in list(C.iter_target_linears(model, backend, args.layer_type)):
        if not weight_only and name not in cov:
            continue
        linalg_device = device if args.linalg_device == "cuda" else "cpu"
        W = mod.weight.data.to(device=linalg_device, dtype=torch.float32)
        stat = None if weight_only else cov[name].to(linalg_device)
        out_dim, in_dim = W.shape
        k = C.rank_from_ratio(args.ratio, out_dim, in_dim)
        if not C.is_compression_beneficial(k, out_dim, in_dim):
            n_skip += 1
            continue
        A, B, k = whiten_truncate(W, stat, args.ratio, decomp=args.decomp)

        prefix = name.replace(".", "_")
        factor_dtype = torch.float32 if args.save_dtype == "float32" else torch.bfloat16
        torch.save(A.to(dtype=factor_dtype).cpu(),
                   os.path.join(args.save_path, f"{prefix}_A.pt"))
        torch.save(B.to(dtype=factor_dtype).cpu(),
                   os.path.join(args.save_path, f"{prefix}_B.pt"))

        parent, attr = C.get_parent_attr(model, name)
        # A/B are already saved to disk; the compressed layer is NOT needed on GPU
        # (no PPL eval follows). Keep the replacement on CPU so the GPU stays flat at
        # ~model size during the SVD phase instead of accumulating 224 layers -> OOM.
        bias = mod.bias.detach().cpu() if mod.bias is not None else None
        setattr(parent, attr, C.LowRankLinear(A.cpu(), B.cpu(), bias=bias))

        orig_params += out_dim * in_dim
        comp_params += out_dim * k + k * in_dim
        n_comp += 1
        summary_layers[name] = {"out": out_dim, "in": in_dim, "k": int(k)}
        # memory hygiene: drop refs, do NOT move 8B params to CPU before del
        del W, stat, A, B
        torch.cuda.empty_cache()

    C.assert_head_dense(model)  # lm_head hard guard AFTER replacement
    print(f"[compress] compressed={n_comp} skipped={n_skip} "
          f"({time.time()-t_dec:.0f}s)")

    # ── summary ───────────────────────────────────────────────────────────────
    peak = C.peak_rss_gb()
    weights_sha256, n_factor_files = fingerprint_weight_dir(args.save_path)
    summary = {
        "arm": arm,
        "calib_scheme": blob.get("scheme", arm),
        "backend": backend.name,
        "model_id": backend.model_id,
        "calib_file": os.path.basename(args.calib) if args.calib else None,
        "calib_git_hash": blob.get("git_hash"),
        "calib_sha256": sha256_file(args.calib) if args.calib else None,
        "calib_objective": blob.get(
            "objective",
            "weight_only_svd" if weight_only else "uniform_activation_reconstruction",
        ),
        "calibration_batch_size": args.batch_size,
        "ratio": args.ratio,
        "layer_type": args.layer_type,
        "decomp": args.decomp,
        "model_path": model_path,
        "n_target_linears": n_targets,
        "n_compressed": n_comp,
        "n_skipped": n_skip,
        "orig_params": orig_params,
        "compressed_params": comp_params,
        "kept_fraction": (comp_params / orig_params) if orig_params else None,
        "ratio_semantics": "ratio = target parameter RETENTION fraction "
                           "(k=int(ratio*out*in/(out+in))); kept_fraction is the "
                           "MEASURED compressed/orig ratio -- trust this, not the symbol",
        "per_layer": summary_layers,
        "git_hash": ghash,
        "peak_rss_gb": peak,
        "wall_clock_sec": time.time() - t0,
        "save_path": args.save_path,
        "run_id": args.run_id,
        "linalg_device": args.linalg_device,
        "model_offloaded_before_linalg": args.offload_model_before_linalg,
        "factor_file_dtype": args.save_dtype,
        "weights_sha256": weights_sha256,
        "n_factor_files": n_factor_files,
    }
    # written both next to weights and into results/compress/
    C.dump_json(summary, os.path.join(args.save_path, "compression_summary.json"))
    artifact_id = args.run_id or ghash
    res_name = f"{artifact_id}_{arm}_{args.layer_type}_r{args.ratio}_summary.json"
    C.dump_json(
        summary,
        C.repo_root() / "results" / "compress" / backend.name / res_name,
    )
    print(f"[compress] summary -> {args.save_path}/compression_summary.json")
    print(f"[compress] peak_rss={peak:.1f}GB  wall={ (time.time()-t0)/60:.1f}min  "
          f"kept={summary['kept_fraction']}")


if __name__ == "__main__":
    main()

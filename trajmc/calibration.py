"""Build clean-t0 or random-t calibration inputs for either backend.

This is the ONLY module where BASE and OURS differ. The difference is a single
switch: whether calibration tokens are noised.

  BASE  (--arm base): t=0, no masking. Clean continuous 2048-token windows.
  OURS  (--arm ours): per window t~U[0,1]; each token masked w.p. t -> MASK_ID.
                      No prefix protection (continuous windows, no prompt/answer).

Window construction is IDENTICAL and deterministic across arms: the same seed
produces byte-identical pre-noise windows, so OURS only ever *overwrites* masked
positions. This is what makes the ③(c) hash/byte-identity check pass.

Corpus:
  --corpus c4      : pure C4-en (main experiment).
  --corpus c4_cot  : C4:CoT = 1:1, CoT from GSM8K TRAIN split only (ablation;
                     not run until the main result is in).

Outputs (to ``results/calib/<backend>/`` by default):
  <prefix>_calib.pt        : dict{input_ids: LongTensor[N, seqlen], windows_pre: ...}
  <prefix>_manifest.json   : full provenance (see below).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import common as C


def build_windows(tokenizer, traindata, nsamples, seqlen, seed):
    """
    Deterministic continuous windows. Uses ONLY the window RNG (seed); no noise.
    Returns LongTensor[nsamples, seqlen] and per-window pre-noise hashes.
    Identical output for base and ours given the same (seed, nsamples, seqlen).
    """
    rng = np.random.default_rng(seed)  # window-selection stream (arm-independent)
    windows = []
    hashes = []
    attempts = 0
    n_docs = len(traindata)
    while len(windows) < nsamples:
        attempts += 1
        if attempts > nsamples * 500:
            raise RuntimeError(
                f"only built {len(windows)}/{nsamples} windows after {attempts} "
                "attempts; the corpus may not contain enough long documents"
            )
        idx = int(rng.integers(0, n_docs))
        enc = tokenizer(traindata[idx]["text"], return_tensors="pt", truncation=False)[
            "input_ids"
        ]
        if enc.shape[1] <= seqlen:
            continue
        start = int(rng.integers(0, enc.shape[1] - seqlen))
        chunk = enc[0, start : start + seqlen].clone().long()
        windows.append(chunk)
        hashes.append(C.sha256_ids(chunk))
    return torch.stack(windows, dim=0), hashes


def apply_noise(windows, seed, mask_id):
    """
    OURS noising. Separate RNG stream (does not perturb window selection).
    Per window t~U[0,1]; token masked w.p. t. Returns (noised, t_list, stats).
    """
    gen = torch.Generator().manual_seed(seed)
    N, L = windows.shape
    t = torch.rand(N, generator=gen).tolist()  # per-window mask probability
    noised = windows.clone()
    measured = []
    thirds = []  # (front, mid, back) mask fraction per window -> positional bias check
    third = L // 3
    for i in range(N):
        r = float(t[i])
        m = torch.rand(L, generator=gen) < r
        noised[i][m] = mask_id
        measured.append(float(m.float().mean()))
        thirds.append(
            [
                float(m[:third].float().mean()),
                float(m[third : 2 * third].float().mean()),
                float(m[2 * third :].float().mean()),
            ]
        )
    return noised, t, measured, thirds


def build_windows_streaming(tokenizer, stream, nsamples, seqlen, seed, max_docs=200000):
    """
    Deterministic continuous windows from a STREAMING dataset (e.g. multilingual
    mC4, which is too large to index). Iterates the stream in its fixed order;
    for each doc long enough, takes one seeded-random seqlen window. Same seed +
    same stream order => byte-identical windows across arms (base/ours).
    """
    rng = np.random.default_rng(seed)
    windows, hashes = [], []
    seen = 0
    for ex in stream:
        seen += 1
        if seen > max_docs:
            raise RuntimeError(
                f"streamed {seen} documents, but only built "
                f"{len(windows)}/{nsamples} windows"
            )
        enc = tokenizer(ex["text"], return_tensors="pt", truncation=False)["input_ids"]
        if enc.shape[1] <= seqlen:
            continue
        start = int(rng.integers(0, enc.shape[1] - seqlen))
        chunk = enc[0, start : start + seqlen].clone().long()
        windows.append(chunk)
        hashes.append(C.sha256_ids(chunk))
        if len(windows) >= nsamples:
            break
    return torch.stack(windows, dim=0), hashes


def load_corpus(corpus, c4_split, cot_split, c4_streaming=False):
    """Return list-like of {'text': ...} plus a composition record."""
    from datasets import load_dataset

    comp = {}
    if corpus == "multilingual":
        # true "不限语种": ROUND-ROBIN interleave of many per-language C4 streams.
        # NB: the 'multilingual' config streams languages sequentially (af, am, ...),
        # so taking the first N windows would be one language only -> we interleave
        # explicit per-language streams to get a genuine mix.
        from datasets import interleave_datasets

        langs = ["en", "zh", "ru", "de", "fr", "es", "ja", "ar", "hi",
                 "pt", "it", "ko", "nl", "tr", "vi", "id"]
        streams = [load_dataset("allenai/c4", l, split="train", streaming=True)
                   .select_columns(["text"])  # align features (timestamp types differ)
                   for l in langs]
        ds = interleave_datasets(streams)  # round-robin across languages
        comp = {"c4": {"dataset": "allenai/c4", "config": "multilingual_interleaved",
                       "langs": langs, "split": "train(streaming)", "count": "streamed"},
                "cot": {"dataset": None, "split": None, "count": 0}}
        return ds, comp, None
    if corpus == "c4":
        split = "train" if c4_streaming else c4_split
        c4 = load_dataset("allenai/c4", "en", split=split,
                          streaming=c4_streaming)
        comp = {"c4": {"dataset": "allenai/c4", "config": "en", "split": split,
                       "streaming": c4_streaming,
                       "count": "streamed" if c4_streaming else len(c4)},
                "cot": {"dataset": None, "split": None, "count": 0}}
        return c4, comp, None
    elif corpus == "c4_cot":
        c4 = load_dataset("allenai/c4", "en", split=c4_split)
        # CoT strictly from GSM8K TRAIN (never test).
        assert "train" in cot_split, "CoT must come from GSM8K TRAIN split only."
        gsm = load_dataset("gsm8k", "main", split=cot_split)
        comp = {
            "c4": {"dataset": "allenai/c4", "config": "en", "split": c4_split,
                   "count": len(c4)},
            "cot": {"dataset": "gsm8k", "config": "main", "split": cot_split,
                    "count": len(gsm)},
        }
        return c4, comp, gsm
    else:
        raise ValueError(f"unknown corpus {corpus}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(C.BACKENDS), required=True)
    ap.add_argument("--arm", choices=["base", "ours"], required=True)
    ap.add_argument("--corpus", choices=["c4", "c4_cot", "multilingual"], default="c4")
    ap.add_argument("--nsamples", type=int, default=None,
                    help="defaults to the backend's experiment setting")
    ap.add_argument("--seqlen", type=int, default=C.SEQLEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_path", type=str, default=None,
                    help="local checkpoint or Hugging Face id")
    ap.add_argument("--c4_split", type=str, default="train[:100000]")
    ap.add_argument("--c4_streaming", action="store_true",
                    help="stream C4 instead of materializing all backing shards; "
                         "the first qualifying documents are deterministic")
    ap.add_argument("--cot_split", type=str, default="train")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="build a tiny set + rich stats, do NOT save the big tensor")
    ap.add_argument("--dry_n", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    backend = C.get_backend(args.backend)
    model_path = args.model_path or backend.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.mask_token_id not in (None, backend.mask_id):
        raise ValueError(
            f"mask id mismatch: configured={backend.mask_id}, "
            f"tokenizer={tokenizer.mask_token_id}"
        )

    configured_nsamples = args.nsamples or backend.default_nsamples
    nsamples = args.dry_n if args.dry_run else configured_nsamples
    out_dir = args.out_dir or str(C.repo_root() / "results" / "calib" / backend.name)

    ghash = C.git_hash()
    tag = "dry" if args.dry_run else "full"
    prefix = (
        f"{backend.name}_{args.arm}_{args.corpus}_n{nsamples}_s{args.seed}_{tag}_{ghash}"
    )

    print(f"[calib] backend={backend.name} arm={args.arm} "
          f"corpus={args.corpus} nsamples={nsamples} "
          f"seqlen={args.seqlen} seed={args.seed}")

    traindata, comp, cot = load_corpus(args.corpus, args.c4_split, args.cot_split,
                                      c4_streaming=args.c4_streaming)

    # ── window construction (identical across arms) ───────────────────────────
    if args.corpus == "multilingual":
        windows, hashes = build_windows_streaming(tokenizer, traindata, nsamples,
                                                  args.seqlen, args.seed)
        cot_used = 0
    elif args.corpus == "c4":
        builder = build_windows_streaming if args.c4_streaming else build_windows
        windows, hashes = builder(tokenizer, traindata, nsamples, args.seqlen,
                                  args.seed)
        cot_used = 0
    else:
        # 1:1 interleave. Build C4 windows and CoT windows separately, then splice.
        n_c4 = (nsamples + 1) // 2
        n_cot = nsamples - n_c4
        w_c4, h_c4 = build_windows(tokenizer, traindata, n_c4, args.seqlen, args.seed)
        # CoT windows: concatenate q+a, pad/truncate to seqlen (separate builder omitted
        # for main run; ablation wiring lives here but is not exercised yet).
        cot_texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in cot]
        w_cot, h_cot = build_windows(
            tokenizer, [{"text": t} for t in cot_texts], n_cot, args.seqlen,
            args.seed + 1,
        )
        windows = torch.cat([w_c4, w_cot], dim=0)
        hashes = h_c4 + h_cot
        cot_used = n_cot

    # ── the ONLY arm difference: noise ────────────────────────────────────────
    if args.arm == "base":
        input_ids = windows.clone()
        t_list = [0.0] * nsamples
        measured = [0.0] * nsamples
        thirds = [[0.0, 0.0, 0.0]] * nsamples
        n_mask_total = int((input_ids == backend.mask_id).sum())
    else:
        input_ids, t_list, measured, thirds = apply_noise(
            windows, args.seed, backend.mask_id
        )
        n_mask_total = int((input_ids == backend.mask_id).sum())

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "arm": args.arm,
        "backend": backend.name,
        "model_id": backend.model_id,
        "noise_mode": "clean_t0" if args.arm == "base" else "uniform_t_U01",
        "corpus": args.corpus,
        "corpus_composition": comp,
        "cot_windows_used": cot_used,
        "nsamples": nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "mask_id": backend.mask_id,
        "git_hash": ghash,
        "model_path": model_path,
        "window_hashes_prenoise": hashes,
        "per_sample_t": t_list,
        "per_sample_measured_mask_ratio": measured,
        "per_sample_mask_thirds": thirds,
        "total_mask_tokens": n_mask_total,
        "total_tokens": int(input_ids.numel()),
        "dry_run": args.dry_run,
    }
    man_path = str(Path(out_dir) / f"{prefix}_manifest.json")
    C.dump_json(manifest, man_path)
    print(f"[calib] manifest -> {man_path}")

    if not args.dry_run:
        calib_path = str(Path(out_dir) / f"{prefix}_calib.pt")
        torch.save({"input_ids": input_ids, "windows_pre": windows,
                    "window_hashes": hashes, "git_hash": ghash,
                    "backend": backend.name, "arm": args.arm,
                    "seed": args.seed}, calib_path)
        print(f"[calib] tensor -> {calib_path}  shape={tuple(input_ids.shape)}")
    else:
        # dry summary for ③ checks
        import numpy as _np
        meas = _np.array(measured)
        tt = _np.array(t_list)
        print(f"[dry] mask ratio mean={meas.mean():.4f}  "
              f"t mean={tt.mean():.4f}  total_mask={n_mask_total}")
        print(f"[dry] window hashes (first 3): {hashes[:3]}")


if __name__ == "__main__":
    main()

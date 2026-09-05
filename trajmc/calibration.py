"""Build matched calibration inputs for either supported backend.

The timestep-distribution ablation exposes five schemes over byte-identical
clean windows: clean t0, iid random-t, deterministic grid-t, grid-t with a
visible prefix, and true closed-loop reverse-rollout states.  Legacy
``--arm base|ours`` remains an alias for ``clean_t0|random_t``.

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
from . import prompts as P
from . import sampling as S


def artifact_prefix(
    backend_name: str,
    label: str,
    scheme: str,
    corpus: str,
    nsamples: int,
    seed: int,
    sampling_seed: int,
    dry_run: bool,
    prefix_ratio: float,
    rollout_steps: int,
    git_hash: str | None = None,
    rollout_prompts: str = "c4",
    rollout_block_length: int | None = None,
) -> str:
    """File-name stem for one calibration artifact.

    This is the only place the name is constructed.  ``--print_artifact_path``
    exposes it so callers never re-derive it: a shell script that guessed the
    scheme or seed suffix wrong would otherwise fail *after* paying for the
    rollout that produced the file.
    """
    tag = "dry" if dry_run else "full"
    seed_tag = (
        f"s{seed}" if sampling_seed == seed
        else f"ws{seed}_ms{sampling_seed}"
    )
    scheme_tag = ""
    if scheme == "rollout" and rollout_prompts != "c4":
        # Task-prompt rollouts have no C4 prefix ratio; their geometry is the
        # deployed prompt plus the deployed block schedule.
        scheme_tag += f"_{rollout_prompts}"
        if rollout_block_length:
            scheme_tag += f"_b{rollout_block_length}"
        scheme_tag += f"_steps{rollout_steps}"
    else:
        if scheme in {"grid_t_prefix", "rollout"}:
            scheme_tag += f"_p{prefix_ratio:g}".replace(".", "p")
        if scheme == "rollout":
            scheme_tag += f"_steps{rollout_steps}"
    ghash = C.git_hash() if git_hash is None else git_hash
    return (
        f"{backend_name}_{label}{scheme_tag}_{corpus}_n{nsamples}_"
        f"{seed_tag}_{tag}_{ghash}"
    )


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


def build_windows_packed(tokenizer, texts, nsamples, seqlen, seed):
    """Windows packed from documents *shorter* than ``seqlen``.

    ``build_windows`` samples a contiguous slice out of one long document and
    skips anything at or below ``seqlen``.  A GSM8K train question plus its
    worked answer is about 187 tokens, so that builder can never emit a window
    from it.  Here documents are shuffled once by the window seed and
    concatenated in order until each window is full, which is the ordinary way
    short supervised examples are packed.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    windows, hashes, buffer, cursor = [], [], [], 0
    while len(windows) < nsamples:
        if cursor >= len(order):
            if not buffer:
                raise RuntimeError(
                    f"only built {len(windows)}/{nsamples} packed windows; the "
                    "corpus does not contain enough text"
                )
            cursor = 0
            order = rng.permutation(len(texts))
        ids = tokenizer(texts[int(order[cursor])], return_tensors="pt",
                        truncation=False)["input_ids"][0]
        cursor += 1
        buffer.append(ids)
        total = sum(int(chunk.shape[0]) for chunk in buffer)
        while total >= seqlen and len(windows) < nsamples:
            joined = torch.cat(buffer)[:seqlen].clone().long()
            windows.append(joined)
            hashes.append(C.sha256_ids(joined))
            leftover = torch.cat(buffer)[seqlen:]
            buffer = [leftover] if leftover.numel() else []
            total = int(leftover.numel())
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
    ap.add_argument(
        "--arm",
        choices=sorted(S.LEGACY_ARM_SCHEMES),
        default=None,
        help="legacy alias: base=clean_t0, ours=random_t",
    )
    ap.add_argument("--scheme", choices=S.SCHEMES, default=None)
    ap.add_argument("--corpus", choices=["c4", "c4_cot", "multilingual"], default="c4")
    ap.add_argument("--nsamples", type=int, default=None,
                    help="defaults to the backend's experiment setting")
    ap.add_argument("--seqlen", type=int, default=C.SEQLEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sampling_seed",
        type=int,
        default=None,
        help="timestep/mask/rollout RNG seed; defaults to --seed",
    )
    ap.add_argument("--model_path", type=str, default=None,
                    help="local checkpoint or Hugging Face id")
    ap.add_argument("--c4_split", type=str, default="train[:100000]")
    ap.add_argument("--c4_streaming", action="store_true",
                    help="stream C4 instead of materializing all backing shards; "
                         "the first qualifying documents are deterministic")
    ap.add_argument("--cot_split", type=str, default="train")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument(
        "--prefix_ratio",
        type=float,
        default=0.25,
        help="visible prefix for grid_t_prefix and rollout",
    )
    ap.add_argument(
        "--rollout_steps",
        type=int,
        default=None,
        help="number of native reverse-sampler model calls",
    )
    ap.add_argument("--rollout_device", default="cuda")
    ap.add_argument("--rollout_prompts", default="c4",
                    help="prompt source for the Actual-State rollout: 'c4' "
                         "(continue calibration text) or a task source such as "
                         "gsm8k_train / gsm8k_test / math500")
    ap.add_argument("--rollout_gen_length", type=int, default=None,
                    help="defaults to the harness's setting for the source")
    ap.add_argument("--rollout_block_length", type=int, default=None,
                    help="defaults to the harness's setting for the source")
    ap.add_argument("--rollout_num_fewshot", type=int, default=None,
                    help="defaults to the harness's setting for the source")
    ap.add_argument("--rollout_prompt_length", type=int, default=None,
                    help="exact prompt token length; defaults to the shortest "
                         "natural few-shot length so no row gains an extra shot")
    ap.add_argument("--rollout_no_chat_template", action="store_true")
    ap.add_argument("--allow_sampler_mismatch", action="store_true",
                    help="permit generation/block/steps that differ from the "
                         "evaluation harness's settings")
    ap.add_argument("--allow_degenerate_rollout", action="store_true",
                    help="record a rollout whose sampler flooded the suffix "
                         "with one token instead of generating text")
    ap.add_argument("--print_artifact_path", action="store_true",
                    help="print the .pt path this exact argv would write, then "
                         "exit without loading the tokenizer, corpus, or model")
    ap.add_argument("--dry_run", action="store_true",
                    help="build a tiny set + rich stats, do NOT save the big tensor")
    ap.add_argument("--dry_n", type=int, default=16)
    args = ap.parse_args()
    # Remember whether --rollout_steps was left to the harness default.
    args.rollout_steps_default_used = args.rollout_steps is None
    if args.rollout_steps is None:
        args.rollout_steps = 256

    backend = C.get_backend(args.backend)
    try:
        scheme, label = S.resolve_scheme(args.arm, args.scheme)
    except ValueError as exc:
        ap.error(str(exc))
    if not 0.0 <= args.prefix_ratio < 1.0:
        ap.error("--prefix_ratio must lie in [0, 1)")
    sampling_seed = args.seed if args.sampling_seed is None else args.sampling_seed
    configured_nsamples = args.nsamples or backend.default_nsamples
    nsamples = args.dry_n if args.dry_run else configured_nsamples
    out_dir = args.out_dir or str(C.repo_root() / "results" / "calib" / backend.name)
    ghash = C.git_hash()
    # Resolve the harness's sampler settings before the name is built, so the
    # block length lands in the file name.  Two runs that differ only in block
    # length must not collide.
    if scheme == "rollout" and args.rollout_prompts != "c4":
        _official = P.official_sampler(args.rollout_prompts)
        if args.rollout_gen_length is None:
            args.rollout_gen_length = _official["gen_length"]
        if args.rollout_block_length is None:
            args.rollout_block_length = _official["block_length"]
        if args.rollout_steps_default_used:
            args.rollout_steps = _official["steps"]
    prefix = artifact_prefix(
        backend.name, label, scheme, args.corpus, nsamples, args.seed,
        sampling_seed, args.dry_run, args.prefix_ratio, args.rollout_steps,
        git_hash=ghash, rollout_prompts=args.rollout_prompts,
        rollout_block_length=args.rollout_block_length,
    )

    # Resolved before the tokenizer, the corpus, and the model: a caller that
    # only wants the path must not pay for a rollout to learn it.
    if args.print_artifact_path:
        print(str(Path(out_dir) / f"{prefix}_calib.pt"))
        return

    from transformers import AutoTokenizer

    model_path = args.model_path or backend.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.mask_token_id not in (None, backend.mask_id):
        raise ValueError(
            f"mask id mismatch: configured={backend.mask_id}, "
            f"tokenizer={tokenizer.mask_token_id}"
        )

    print(f"[calib] backend={backend.name} scheme={scheme} label={label} "
          f"corpus={args.corpus} nsamples={nsamples} "
          f"seqlen={args.seqlen} window_seed={args.seed} "
          f"sampling_seed={sampling_seed}")

    # The Actual-State control's obligation is to be close to deployment, not
    # to share a corpus with the matched arms.  For a mathematical-reasoning
    # study its states come from solving maths problems.
    task_rollout = scheme == "rollout" and args.rollout_prompts != "c4"
    prompt_meta = {}
    if task_rollout:
        # The harness's own settings, so a hand-passed generation or block
        # length cannot silently disagree with deployment.
        official = P.official_sampler(args.rollout_prompts)
        mismatch = {
            key: (value, official[key])
            for key, value in (("gen_length", args.rollout_gen_length),
                               ("block_length", args.rollout_block_length),
                               ("steps", args.rollout_steps))
            if value != official[key]
        }
        if mismatch and not args.allow_sampler_mismatch:
            ap.error(
                f"sampler settings differ from the harness for "
                f"{args.rollout_prompts}: {mismatch}. Pass "
                f"--allow_sampler_mismatch to override deliberately."
            )
        prompts, prompt_meta = P.build_task_prompts(
            tokenizer,
            args.rollout_prompts,
            nsamples,
            args.rollout_prompt_length,
            args.seed,
            chat_template=not args.rollout_no_chat_template,
            num_fewshot=args.rollout_num_fewshot,
        )
        prompt_length = prompts.shape[1]
        # The "pre" state is the real pre-generation state: prompt plus an
        # all-MASK generation region.  There is no ground-truth continuation.
        windows = torch.cat(
            [
                prompts,
                torch.full(
                    (nsamples, args.rollout_gen_length),
                    backend.mask_id,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
        hashes = [C.sha256_ids(row) for row in prompts]
        comp, cot_used, traindata = dict(prompt_meta), 0, None
        print(f"[calib] task prompts: {args.rollout_prompts} "
              f"prompt_length={prompt_length} "
              f"gen_length={args.rollout_gen_length} "
              f"block_length={args.rollout_block_length} "
              f"fewshot={prompt_meta['prompt_num_fewshot']} "
              f"fixed_shots={prompt_meta['prompt_fewshot_fixed']}")
    else:
        traindata, comp, cot = load_corpus(args.corpus, args.c4_split, args.cot_split,
                                          c4_streaming=args.c4_streaming)

    # ── window construction (identical across arms) ───────────────────────────
    if task_rollout:
        pass
    elif args.corpus == "multilingual":
        windows, hashes = build_windows_streaming(tokenizer, traindata, nsamples,
                                                  args.seqlen, args.seed)
        cot_used = 0
    elif args.corpus == "c4":
        builder = build_windows_streaming if args.c4_streaming else build_windows
        windows, hashes = builder(tokenizer, traindata, nsamples, args.seqlen,
                                  args.seed)
        cot_used = 0
    else:
        # 1:1 interleave.  The C4 half uses the same builder and seed as the
        # pure-C4 arms, so its windows are the pure-C4 arm's first n_c4 windows
        # and the ablation isolates the corpus, not the window sampler.
        n_c4 = (nsamples + 1) // 2
        n_cot = nsamples - n_c4
        builder = build_windows_streaming if args.c4_streaming else build_windows
        w_c4, h_c4 = builder(tokenizer, traindata, n_c4, args.seqlen, args.seed)
        cot_texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in cot]
        w_cot, h_cot = build_windows_packed(
            tokenizer, cot_texts, n_cot, args.seqlen, args.seed + 1
        )
        windows = torch.cat([w_c4, w_cot], dim=0)
        hashes = h_c4 + h_cot
        cot_used = n_cot

    # ── calibration-state construction; clean windows are identical ──────────
    rollout_metadata = {}
    if scheme == "clean_t0":
        input_ids = windows.clone()
        t_list = [0.0] * nsamples
        prefix_length = 0
    elif scheme == "random_t":
        input_ids, t_list, measured, thirds = apply_noise(
            windows, sampling_seed, backend.mask_id
        )
        prefix_length = 0
    else:
        timesteps = S.uniform_grid(nsamples)
        t_list = timesteps.tolist()
        prefix_ratio = args.prefix_ratio if scheme in {
            "grid_t_prefix", "rollout"
        } else 0.0
        prefix_length = int(prefix_ratio * windows.shape[1])
        if scheme in {"grid_t", "grid_t_prefix"}:
            input_ids = S.apply_forward_mask(
                windows,
                timesteps,
                backend.mask_id,
                seed=sampling_seed,
                prefix_ratio=prefix_ratio,
            )
        elif scheme == "rollout":
            if not torch.cuda.is_available() and args.rollout_device.startswith("cuda"):
                ap.error("rollout calibration requested CUDA, but CUDA is unavailable")
            model, _ = C.load_model(
                backend,
                model_path=model_path,
                dtype=(
                    torch.bfloat16
                    if args.rollout_device.startswith("cuda")
                    else torch.float32
                ),
                device=args.rollout_device,
            )
            if task_rollout:
                prefix_length = prompt_length
                input_ids, rollout_metadata = S.llada_block_rollout_states(
                    model,
                    prompts,
                    timesteps,
                    backend.mask_id,
                    args.rollout_gen_length,
                    args.rollout_block_length,
                    args.rollout_steps,
                    logits_eos_inf=official["logits_eos_inf"],
                    confidence_eos_eot_inf=official["confidence_eos_eot_inf"],
                )
                rollout_metadata.update(prompt_meta)
            else:
                input_ids, rollout_metadata = S.rollout_states(
                    C.arch_family(backend),
                    model,
                    windows,
                    timesteps,
                    backend.mask_id,
                    prefix_ratio,
                    args.rollout_steps,
                    sampling_seed,
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eos_ids = sorted({
                identifier for identifier in (
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                )
                if isinstance(identifier, int) and identifier >= 0
            })
            rollout_metadata.update(
                S.revealed_token_diagnostics(
                    input_ids, backend.mask_id, prefix_length,
                    eos_token_ids=eos_ids,
                )
            )
            rollout_metadata["eos_token_ids"] = eos_ids
            rollout_metadata["decoded_samples"] = S.decode_revealed_samples(
                input_ids, tokenizer, backend.mask_id, prefix_length, count=5
            )
        else:  # pragma: no cover - argparse/resolve_scheme guard this
            raise AssertionError(scheme)

    stats = S.mask_statistics(input_ids, backend.mask_id, prefix_length)
    measured = stats["per_sample_measured_mask_ratio"]
    thirds = stats["per_sample_mask_thirds"]
    n_mask_total = stats["total_mask_tokens"]
    if scheme != "rollout":
        visible = input_ids.ne(backend.mask_id)
        if not torch.equal(input_ids[visible], windows[visible]):
            raise RuntimeError("offline corruption changed a visible clean token")
    if prefix_length and not torch.equal(
        input_ids[:, :prefix_length], windows[:, :prefix_length]
    ):
        raise RuntimeError("visible-prefix identity invariant failed")

    revealed_suffix = input_ids[:, prefix_length:].ne(backend.mask_id)
    prediction_mismatches = int(
        (
            revealed_suffix
            & input_ids[:, prefix_length:].ne(windows[:, prefix_length:])
        ).sum()
    )
    revealed_suffix_count = int(revealed_suffix.sum())

    objective = {
        "clean_t0": "activation_reconstruction_on_clean_t0",
        "random_t": "iid_uniform_t_forward_corruption",
        "grid_t": "uniform_grid_t_forward_corruption",
        "grid_t_prefix": "uniform_grid_t_prefix_preserving_forward_corruption",
        "rollout": "uniform_exposure_real_reverse_sampler_states",
    }[scheme]

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "arm": label,
        "scheme": scheme,
        "objective": objective,
        "backend": backend.name,
        "model_id": backend.model_id,
        "noise_mode": scheme,
        "corpus": args.corpus,
        "corpus_composition": comp,
        "cot_windows_used": cot_used,
        "nsamples": nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "window_seed": args.seed,
        "sampling_seed": sampling_seed,
        "mask_id": backend.mask_id,
        "git_hash": ghash,
        "model_path": model_path,
        "window_hashes_prenoise": hashes,
        "per_sample_t": t_list,
        "per_sample_measured_mask_ratio": measured,
        "per_sample_measured_suffix_mask_ratio": stats[
            "per_sample_measured_suffix_mask_ratio"
        ],
        "per_sample_mask_thirds": thirds,
        "total_mask_tokens": n_mask_total,
        "total_tokens": int(input_ids.numel()),
        "prefix_ratio": prefix_length / windows.shape[1],
        "prefix_length": prefix_length,
        "suffix_length": windows.shape[1] - prefix_length,
        "timestep_design": (
            "iid_uniform" if scheme == "random_t" else
            "deterministic_uniform_grid" if scheme in {
                "grid_t", "grid_t_prefix", "rollout"
            } else "endpoint_t0"
        ),
        "model_prediction_feedback": scheme == "rollout",
        "revealed_suffix_tokens": revealed_suffix_count,
        "prediction_mismatches_vs_clean": (
            None if task_rollout else prediction_mismatches
        ),
        "prediction_mismatch_fraction": (
            None if task_rollout
            else prediction_mismatches / max(revealed_suffix_count, 1)
        ),
        "prediction_mismatch_caveat": (
            "ambiguous by construction: 'generated different text' and "
            "'generated nothing' both approach 1.0. Never judge rollout health "
            "from this field alone -- use the revealed-token diagnostics."
        ),
        "rollout_steps": args.rollout_steps if scheme == "rollout" else None,
        "dry_run": args.dry_run,
        **rollout_metadata,
    }
    man_path = str(Path(out_dir) / f"{prefix}_manifest.json")
    C.dump_json(manifest, man_path)
    print(f"[calib] manifest -> {man_path}")

    if scheme == "rollout":
        share = manifest.get("top_revealed_token_share")
        if share is None:
            print("[calib] rollout revealed no tokens")
        else:
            eos_share = manifest.get("eos_revealed_share")
            print(f"[calib] rollout diagnostics: "
                  f"unique={manifest['unique_revealed_tokens']} "
                  f"ratio={manifest['unique_revealed_ratio']:.5f} "
                  f"top={manifest['top_revealed_token_id']}@{share:.2%} "
                  f"eos={'n/a' if eos_share is None else f'{eos_share:.2%}'} "
                  f"median_per_row={manifest['median_distinct_tokens_per_row']}")
        # Thresholds catch the failure modes we know about; reading the text is
        # what catches the ones we do not.  Always print, never optional.
        for sample in manifest.get("decoded_samples", []):
            print(f"[calib]   --- row {sample['row']}, revealed "
                  f"{sample['revealed']}/{sample['of']} ---")
            print(f"[calib]   prompt tail: {sample['prompt_tail']!r}")
            print(f"[calib]   generated  : {sample['generated']!r}")
        # The manifest is already on disk for postmortem; refuse to save an
        # Actual-State artifact that the sampler never actually generated.
        if not args.allow_degenerate_rollout:
            S.assert_rollout_not_degenerate(manifest)

    if not args.dry_run:
        calib_path = str(Path(out_dir) / f"{prefix}_calib.pt")
        blob = dict(manifest)
        blob.update(
            input_ids=input_ids,
            windows_pre=windows,
            clean_ids=windows,
            window_hashes=hashes,
            attention_mask=torch.ones_like(input_ids),
        )
        torch.save(blob, calib_path)
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

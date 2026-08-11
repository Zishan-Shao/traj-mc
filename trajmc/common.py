"""Shared model adapters and utilities for Traj-MC.

The random-t calibration and compression algorithms are architecture agnostic.
Only model identifiers, mask tokens, and target Linear names differ between
LLaDA and Dream.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterator, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BackendSpec:
    """Everything in the core pipeline that is model-family specific."""

    name: str
    model_id: str
    mask_id: int
    default_nsamples: int
    block_marker: str
    attention_suffixes: tuple[str, ...]
    mlp_suffixes: tuple[str, ...]
    expected_linears: int
    tasks: Mapping[str, Mapping[str, object]]

    @property
    def all_suffixes(self) -> tuple[str, ...]:
        return self.attention_suffixes + self.mlp_suffixes


LLADA_TASKS = {
    "arc_challenge": dict(fs=0, cfg=0.5, mc_num=128, batch_size=8,
                          metric="acc", gen=False),
    "arc_easy": dict(fs=0, cfg=0.5, mc_num=128, batch_size=8,
                     metric="acc", gen=False),
    "hellaswag": dict(fs=0, cfg=0.5, mc_num=128, batch_size=8,
                      metric="acc_norm", gen=False),
    "piqa": dict(fs=0, cfg=0.5, mc_num=128, batch_size=8,
                 metric="acc_norm", gen=False),
    "winogrande": dict(fs=5, cfg=0.0, mc_num=128, batch_size=8,
                       metric="acc", gen=False),
    "mmlu": dict(fs=5, cfg=0.0, mc_num=1, batch_size=1,
                 metric="acc", gen=False),
    "gsm8k": dict(fs=5, gen_length=256, steps=256, block_length=256,
                  batch_size=1, metric="exact_match", gen=True),
}


DREAM_TASKS = {
    "arc_challenge": dict(fs=0, batch_size=32, metric="acc_norm", gen=False),
    "arc_easy": dict(fs=0, batch_size=32, metric="acc_norm", gen=False),
    "hellaswag": dict(fs=0, batch_size=32, metric="acc_norm", gen=False),
    "piqa": dict(fs=0, batch_size=32, metric="acc_norm", gen=False),
    "winogrande": dict(fs=5, batch_size=32, metric="acc", gen=False),
    "mmlu": dict(fs=5, batch_size=32, metric="acc", gen=False),
    "gsm8k_cot": dict(
        fs=8,
        batch_size=1,
        metric="exact_match",
        gen=True,
        max_new_tokens=256,
        diffusion_steps=256,
        temperature=0.0,
        top_p=0.95,
    ),
}


BACKENDS = {
    "llada": BackendSpec(
        name="llada",
        model_id="GSAI-ML/LLaDA-8B-Base",
        mask_id=126336,
        default_nsamples=256,
        block_marker=".blocks.",
        attention_suffixes=("q_proj", "k_proj", "v_proj", "attn_out"),
        mlp_suffixes=("ff_proj", "up_proj", "ff_out"),
        expected_linears=224,
        tasks=LLADA_TASKS,
    ),
    "dream": BackendSpec(
        name="dream",
        model_id="Dream-org/Dream-v0-Base-7B",
        mask_id=151666,
        default_nsamples=256,
        block_marker=".layers.",
        attention_suffixes=("q_proj", "k_proj", "v_proj", "o_proj"),
        mlp_suffixes=("gate_proj", "up_proj", "down_proj"),
        expected_linears=196,
        tasks=DREAM_TASKS,
    ),
}

SEQLEN = 2048


def get_backend(name: str) -> BackendSpec:
    try:
        return BACKENDS[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown backend {name!r}; choose one of: {choices}") from exc


def primary_metric(backend: BackendSpec | str, task: str) -> str:
    spec = get_backend(backend) if isinstance(backend, str) else backend
    return str(spec.tasks[task]["metric"])


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_hash(short: bool = True) -> str:
    """Git revision used to stamp generated artifacts."""
    try:
        args = ["git", "-C", str(repo_root()), "rev-parse"]
        if short:
            args.append("--short")
        args.append("HEAD")
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def sha256_ids(ids) -> str:
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().to(torch.int64).tolist()
    payload = ",".join(str(int(value)) for value in ids).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_model(
    backend: BackendSpec | str,
    model_path: str | None = None,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """Load a dense model and tokenizer for either supported backend."""
    from transformers import AutoModel, AutoTokenizer

    spec = get_backend(backend) if isinstance(backend, str) else backend
    source = model_path or spec.model_id
    model = AutoModel.from_pretrained(
        source, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.mask_token_id is not None and tokenizer.mask_token_id != spec.mask_id:
        raise ValueError(
            f"{spec.name} mask id mismatch: configured={spec.mask_id}, "
            f"tokenizer={tokenizer.mask_token_id}"
        )
    return model, tokenizer


def is_target_linear(
    name: str,
    module: nn.Module,
    backend: BackendSpec | str,
    layer_type: str = "all",
) -> bool:
    spec = get_backend(backend) if isinstance(backend, str) else backend
    if not isinstance(module, nn.Linear) or spec.block_marker not in name:
        return False
    if layer_type == "all":
        suffixes = spec.all_suffixes
    elif layer_type == "attn":
        suffixes = spec.attention_suffixes
    elif layer_type == "mlp":
        suffixes = spec.mlp_suffixes
    else:
        raise ValueError(f"unknown layer_type {layer_type!r}")
    return name.endswith(suffixes)


def iter_target_linears(
    model: nn.Module,
    backend: BackendSpec | str,
    layer_type: str = "all",
) -> Iterator[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if is_target_linear(name, module, backend, layer_type):
            yield name, module


def count_target_linears(
    model: nn.Module, backend: BackendSpec | str, layer_type: str = "all"
) -> int:
    return sum(1 for _ in iter_target_linears(model, backend, layer_type))


def get_parent_attr(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def rank_from_ratio(ratio: float, out_dim: int, in_dim: int) -> int:
    """SVD-LLM rank formula; ``ratio`` is the target retained fraction."""
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    return max(1, int(ratio * out_dim * in_dim / (out_dim + in_dim)))


def is_compression_beneficial(k: int, out_dim: int, in_dim: int) -> bool:
    return k * (out_dim + in_dim) < out_dim * in_dim


class LowRankLinear(nn.Module):
    """Two Linear layers implementing ``A @ B`` while preserving the bias."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias=None):
        super().__init__()
        k, in_dim = B.shape
        out_dim, _ = A.shape
        self.B = nn.Linear(in_dim, k, bias=False)
        self.A = nn.Linear(k, out_dim, bias=bias is not None)
        self.B.weight.data = B.to(torch.bfloat16)
        self.A.weight.data = A.to(torch.bfloat16)
        if bias is not None:
            self.A.bias.data = bias.to(torch.bfloat16)

    def forward(self, inputs):
        return self.A(self.B(inputs))


def get_output_head(model: nn.Module):
    try:
        head = model.get_output_embeddings()
        if head is not None:
            return head
    except Exception:
        pass
    head = getattr(model, "lm_head", None)
    if head is not None:
        return head
    try:
        return model.model.transformer.ff_out
    except Exception:
        return None


def assert_head_dense(model: nn.Module) -> bool:
    head = get_output_head(model)
    if isinstance(head, (LowRankLinear, nn.Identity)):
        raise RuntimeError(
            f"output head was replaced by {type(head).__name__}; it must stay dense"
        )
    if head is not None and not isinstance(head, (nn.Linear, nn.Embedding)):
        raise RuntimeError(f"unexpected output-head type {type(head).__name__}")
    return True


def dump_json(obj, path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def peak_rss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

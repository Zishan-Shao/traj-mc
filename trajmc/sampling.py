"""Calibration-state samplers for the timestep-distribution ablation.

The three offline schemes independently corrupt a clean sequence.  The rollout
scheme instead records an actual pre-forward state from the dense model's
reverse sampler, so revealed model predictions persist in later states.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMES = (
    "clean_t0",
    "random_t",
    "grid_t",
    "grid_t_prefix",
    "rollout",
)

LEGACY_ARM_SCHEMES = {"base": "clean_t0", "ours": "random_t"}


def resolve_scheme(arm: str | None, scheme: str | None) -> tuple[str, str]:
    """Resolve the legacy BASE/OURS interface and return ``(scheme, label)``."""
    if scheme is None:
        if arm is None:
            raise ValueError("provide --scheme, or legacy --arm base|ours")
        return LEGACY_ARM_SCHEMES[arm], arm
    if scheme not in SCHEMES:
        raise ValueError(f"unknown calibration scheme: {scheme}")
    if arm is not None and LEGACY_ARM_SCHEMES[arm] != scheme:
        raise ValueError(f"--arm {arm} conflicts with --scheme {scheme}")
    return scheme, arm or scheme


def uniform_grid(num_samples: int) -> torch.Tensor:
    """Quant-dLLM release grid: ``1/N, 2/N, ..., N/N``."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return torch.arange(1, num_samples + 1, dtype=torch.float64) / num_samples


def rollout_call_indices(timesteps: torch.Tensor, steps: int) -> torch.Tensor:
    """Map forward noise levels to reverse-sampler pre-forward call indices.

    ``t=1`` is the initial all-MASK state (call 0); values near zero select the
    final pre-forward state.  If ``len(timesteps) == steps`` and timesteps are
    the uniform grid, every sampler call is selected exactly once.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    values = torch.as_tensor(timesteps, dtype=torch.float64)
    if bool(((values <= 0) | (values > 1)).any()):
        raise ValueError("rollout timesteps must lie in (0, 1]")
    return torch.floor((1.0 - values) * steps).long().clamp(0, steps - 1)


def rollout_call_indices_for_transfers(
    timesteps: torch.Tensor, transfers: list[int]
) -> torch.Tensor:
    """Match target mask ratios to states under a non-uniform transfer schedule.

    A call-index grid is only a mask-ratio grid when every sampler call reveals
    the same number of tokens.  LLaDA front-loads the remainder when the suffix
    length is not divisible by ``steps``.  Select the pre-forward state whose
    remaining MASK count is closest to each requested ``t`` instead.
    """
    if not transfers or any(value <= 0 for value in transfers):
        raise ValueError("transfers must be a non-empty list of positive counts")
    values = torch.as_tensor(timesteps, dtype=torch.float64)
    if bool(((values <= 0) | (values > 1)).any()):
        raise ValueError("rollout timesteps must lie in (0, 1]")

    suffix_length = sum(transfers)
    revealed_before_call = torch.tensor(
        [0, *torch.tensor(transfers, dtype=torch.long).cumsum(0)[:-1].tolist()],
        dtype=torch.long,
    )
    remaining = suffix_length - revealed_before_call
    target = values[:, None] * suffix_length
    return (remaining[None, :].double() - target).abs().argmin(dim=1)


def apply_forward_mask(
    windows: torch.Tensor,
    timesteps: torch.Tensor,
    mask_id: int,
    seed: int,
    prefix_ratio: float = 0.0,
) -> torch.Tensor:
    """Independently Bernoulli-mask clean windows conditional on each ``t``."""
    if windows.ndim != 2:
        raise ValueError("windows must have shape [samples, sequence_length]")
    values = torch.as_tensor(timesteps, dtype=torch.float64).cpu()
    if values.numel() != windows.shape[0]:
        raise ValueError("one timestep is required per calibration window")
    if bool(((values < 0) | (values > 1)).any()):
        raise ValueError("timesteps must lie in [0, 1]")
    if not 0.0 <= prefix_ratio < 1.0:
        raise ValueError("prefix_ratio must lie in [0, 1)")

    prefix_length = int(prefix_ratio * windows.shape[1])
    noised = windows.clone()
    for row, timestep in enumerate(values.tolist()):
        # Match Quant-dLLM's released MCS seeding policy: each sample owns an
        # independent stream seeded by ``seed + sample_index``.
        generator = torch.Generator().manual_seed(seed + row)
        mask = torch.rand(windows.shape[1], generator=generator) < timestep
        mask[:prefix_length] = False
        noised[row, mask] = mask_id
    return noised


def mask_statistics(
    states: torch.Tensor, mask_id: int, prefix_length: int = 0
) -> dict[str, object]:
    """Return per-sample full/suffix mask ratios and positional thirds."""
    masks = states.eq(mask_id)
    length = states.shape[1]
    third = length // 3
    suffix_length = length - prefix_length
    full = masks.float().mean(dim=1)
    if suffix_length:
        suffix = masks[:, prefix_length:].float().mean(dim=1)
    else:
        suffix = torch.zeros(states.shape[0])
    thirds = torch.stack(
        (
            masks[:, :third].float().mean(dim=1),
            masks[:, third : 2 * third].float().mean(dim=1),
            masks[:, 2 * third :].float().mean(dim=1),
        ),
        dim=1,
    )
    return {
        "total_mask_tokens": int(masks.sum()),
        "per_sample_measured_mask_ratio": full.tolist(),
        "per_sample_measured_suffix_mask_ratio": suffix.tolist(),
        "per_sample_mask_thirds": thirds.tolist(),
    }


class _SelectedHeadCaptured(RuntimeError):
    pass


class _SelectedHeadProjector:
    """Run the vocabulary projection only at currently masked positions."""

    def __init__(self, model):
        self.model = model
        self.head = model.get_output_embeddings()
        if not isinstance(self.head, nn.Linear):
            raise TypeError("rollout collection requires an nn.Linear output head")
        self.selection = None
        self.hidden = None

        def capture(_module, inputs):
            self.hidden = inputs[0][self.selection].detach()
            raise _SelectedHeadCaptured()

        self.handle = self.head.register_forward_pre_hook(capture)

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor, selection: torch.Tensor):
        self.selection = selection
        self.hidden = None
        try:
            self.model(batch)
        except _SelectedHeadCaptured:
            pass
        if self.hidden is None:
            raise RuntimeError("failed to capture hidden states before the output head")
        logits = F.linear(self.hidden, self.head.weight, self.head.bias)
        if getattr(self.model.config, "scale_logits", False):
            logits.mul_(1.0 / math.sqrt(self.model.config.d_model))
        return logits

    def close(self):
        self.handle.remove()


def _model_device(model) -> torch.device:
    try:
        return torch.device(model.device)
    except (AttributeError, TypeError):
        return next(model.parameters()).device


def _transfer_schedule(suffix_length: int, steps: int) -> list[int]:
    if suffix_length <= 0:
        raise ValueError("rollout suffix must contain at least one token")
    if not 0 < steps <= suffix_length:
        raise ValueError("rollout steps must lie in [1, suffix_length]")
    base, remainder = divmod(suffix_length, steps)
    return [base + int(index < remainder) for index in range(steps)]


@torch.no_grad()
def llada_rollout_states(
    model,
    clean_windows: torch.Tensor,
    timesteps: torch.Tensor,
    mask_id: int,
    prefix_ratio: float,
    steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Collect one true LLaDA pre-forward reverse state per clean window."""
    if clean_windows.ndim != 2:
        raise ValueError("clean_windows must have shape [samples, tokens]")
    if bool(clean_windows.eq(mask_id).any()):
        raise ValueError("clean rollout windows must not already contain MASK")
    prefix_length = int(prefix_ratio * clean_windows.shape[1])
    suffix_length = clean_windows.shape[1] - prefix_length
    transfers = _transfer_schedule(suffix_length, steps)
    calls = rollout_call_indices_for_transfers(timesteps, transfers)
    revealed_before_call = torch.tensor(
        [0, *torch.tensor(transfers).cumsum(0)[:-1].tolist()]
    )
    realized_ratios = (
        (suffix_length - revealed_before_call[calls]).double() / suffix_length
    )
    device = _model_device(model)
    collected = []

    projector = _SelectedHeadProjector(model)
    try:
        for row, target_call in enumerate(calls.tolist()):
            state = torch.full(
                (1, clean_windows.shape[1]),
                mask_id,
                dtype=torch.long,
                device=device,
            )
            state[:, :prefix_length] = clean_windows[
                row : row + 1, :prefix_length
            ].to(device)
            for call in range(target_call):
                selection = state.eq(mask_id)
                positions = torch.nonzero(selection[0], as_tuple=False).squeeze(1)
                logits = projector(state, selection)
                max_logits, predictions = logits.max(dim=-1)
                confidence = torch.exp(max_logits - torch.logsumexp(logits, dim=-1))
                take = min(transfers[call], int(positions.numel()))
                if take:
                    chosen = torch.topk(confidence, k=take).indices
                    state[0, positions[chosen]] = predictions[chosen]
            collected.append(state[0].cpu().clone())
    finally:
        projector.close()

    states = torch.stack(collected)
    return states, {
        "rollout_call_indices": calls.tolist(),
        "rollout_transfer_schedule": transfers,
        "rollout_realized_suffix_mask_ratios": realized_ratios.tolist(),
        "sampler": "llada temperature=0 cfg=0 low_confidence single_block",
    }


class _DreamStateCaptured(RuntimeError):
    pass


@torch.no_grad()
def dream_rollout_states(
    model,
    clean_windows: torch.Tensor,
    timesteps: torch.Tensor,
    mask_id: int,
    prefix_ratio: float,
    steps: int,
    temperature: float = 0.0,
    top_p: float | None = 0.95,
    alg: str = "entropy",
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Collect one true Dream pre-forward reverse state per clean window."""
    if not hasattr(model, "diffusion_generate"):
        raise TypeError("Dream rollout requires model.diffusion_generate")
    if bool(clean_windows.eq(mask_id).any()):
        raise ValueError("clean rollout windows must not already contain MASK")
    prefix_length = int(prefix_ratio * clean_windows.shape[1])
    suffix_length = clean_windows.shape[1] - prefix_length
    if prefix_length <= 0 or suffix_length <= 0:
        raise ValueError("Dream rollout requires a non-empty prefix and suffix")
    calls = rollout_call_indices(timesteps, steps)
    device = _model_device(model)
    collected = []

    for row, target_call in enumerate(calls.tolist()):
        captured = None

        def capture(step, state, _logits):
            nonlocal captured
            call = 0 if step is None else int(step) + 1
            if call == target_call:
                captured = state.detach().cpu().clone()
                raise _DreamStateCaptured()
            return state

        prompt = clean_windows[row : row + 1, :prefix_length].to(device)
        torch.manual_seed(seed + row)
        try:
            model.diffusion_generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=suffix_length,
                return_dict_in_generate=True,
                output_history=False,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=0.0,
                mask_token_id=mask_id,
                generation_tokens_hook_func=capture,
            )
        except _DreamStateCaptured:
            pass
        if captured is None:
            raise RuntimeError(f"Dream sampler did not expose call {target_call}")
        collected.append(captured[0])

    states = torch.stack(collected)
    return states, {
        "rollout_call_indices": calls.tolist(),
        "sampler": (
            f"dream temperature={temperature} top_p={top_p} alg={alg} "
            "alg_temp=0"
        ),
    }


def rollout_states(
    backend: str,
    model,
    clean_windows: torch.Tensor,
    timesteps: torch.Tensor,
    mask_id: int,
    prefix_ratio: float,
    steps: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    if backend == "llada":
        return llada_rollout_states(
            model, clean_windows, timesteps, mask_id, prefix_ratio, steps
        )
    if backend == "dream":
        return dream_rollout_states(
            model,
            clean_windows,
            timesteps,
            mask_id,
            prefix_ratio,
            steps,
            seed=seed,
        )
    raise ValueError(f"unsupported rollout backend: {backend}")

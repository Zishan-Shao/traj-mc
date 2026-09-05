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


def revealed_token_diagnostics(
    states: torch.Tensor,
    mask_id: int,
    prefix_length: int = 0,
    eos_token_ids: "list[int] | None" = None,
) -> dict[str, object]:
    """Concentration of the tokens a rollout actually revealed.

    A reverse rollout is only a usable Actual-State control if the sampler
    produced real text.  A misconfigured one -- a block far longer than the
    deployed block length, a prompt outside the checkpoint's input format --
    fails silently: the model floods the suffix with a single high-confidence
    token, and every downstream statistic is computed on that.

    ``prediction_mismatch_fraction`` cannot be used for this on its own: "wrote
    different text" and "wrote nothing" both drive it to ~1.0.  These
    statistics separate the two.
    """
    suffix = states[:, prefix_length:]
    revealed = suffix.ne(mask_id)
    values = suffix[revealed]
    total = int(values.numel())
    if total == 0:
        return {
            "revealed_tokens": 0,
            "unique_revealed_tokens": 0,
            "unique_revealed_ratio": None,
            "top_revealed_token_id": None,
            "top_revealed_token_share": None,
            "eos_revealed_share": None,
            "median_distinct_tokens_per_row": 0,
        }
    unique, counts = values.unique(return_counts=True)
    top = int(counts.argmax())
    per_row = sorted(
        int(suffix[row][revealed[row]].unique().numel())
        for row in range(suffix.shape[0])
        if bool(revealed[row].any())
    )
    eos_share = None
    if eos_token_ids:
        eos = torch.tensor(sorted(set(int(i) for i in eos_token_ids)))
        eos_share = float(torch.isin(values, eos).sum()) / total
    return {
        "revealed_tokens": total,
        "unique_revealed_tokens": int(unique.numel()),
        "unique_revealed_ratio": int(unique.numel()) / total,
        "top_revealed_token_id": int(unique[top]),
        "top_revealed_token_share": float(counts[top]) / total,
        "eos_revealed_share": eos_share,
        "median_distinct_tokens_per_row": (
            per_row[len(per_row) // 2] if per_row else 0
        ),
    }


def assert_rollout_not_degenerate(
    diagnostics: dict,
    max_top_token_share: float = 0.5,
    max_eos_share: float = 0.5,
    min_unique_ratio: float = 0.01,
    min_median_distinct_per_row: int = 8,
) -> None:
    """Hard gate: refuse a rollout that flooded one token instead of writing text.

    Reference points measured on this repository's own artifacts.  Genuine C4
    text: top token 3.85% of revealed positions, unique ratio 0.108, median 350
    distinct per row.  The collapsed LLaDA-Instruct rollout: top token 99.93%,
    unique ratio 0.00038, median 1.  Every limit below sits an order of
    magnitude away from both.
    """
    if diagnostics.get("revealed_tokens", 0) == 0:
        return
    share = diagnostics.get("top_revealed_token_share")
    eos_share = diagnostics.get("eos_revealed_share")
    unique_ratio = diagnostics.get("unique_revealed_ratio")
    median = diagnostics.get("median_distinct_tokens_per_row", 0)
    problems = []
    if share is not None and share > max_top_token_share:
        problems.append(
            f"token {diagnostics['top_revealed_token_id']} is {share:.2%} of "
            f"revealed tokens (limit {max_top_token_share:.0%})"
        )
    if eos_share is not None and eos_share > max_eos_share:
        problems.append(
            f"EOS/EOT tokens are {eos_share:.2%} of revealed tokens "
            f"(limit {max_eos_share:.0%})"
        )
    if unique_ratio is not None and unique_ratio < min_unique_ratio:
        problems.append(
            f"only {diagnostics['unique_revealed_tokens']} distinct tokens over "
            f"{diagnostics['revealed_tokens']} positions, a ratio of "
            f"{unique_ratio:.5f} (limit {min_unique_ratio})"
        )
    if median < min_median_distinct_per_row:
        problems.append(
            f"the median rollout revealed only {median} distinct token(s) "
            f"(limit {min_median_distinct_per_row})"
        )
    if problems:
        raise RuntimeError(
            "degenerate rollout: " + "; ".join(problems) + ". The sampler did "
            "not generate text. Check block_length against the deployed value "
            "-- LLaDA-8B-Instruct collapses into EOS padding without block "
            "diffusion -- and whether the prompt matches the checkpoint's "
            "expected input format. Pass --allow_degenerate_rollout to record "
            "it anyway."
        )


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


def block_transfer_schedule(
    gen_length: int, block_length: int, steps: int
) -> tuple[list[int], int, int]:
    """Flat per-call reveal counts for LLaDA's block-diffusion sampler.

    Mirrors ``eval/llada_generate.generate``: the generation region is split
    into ``gen_length // block_length`` blocks, each block gets
    ``steps // num_blocks`` sampler calls, and only the current block is a
    candidate for revealing.  Returns the concatenated per-call counts along
    with the block and per-block step counts.
    """
    if gen_length <= 0 or block_length <= 0 or steps <= 0:
        raise ValueError("gen_length, block_length and steps must be positive")
    if gen_length % block_length:
        raise ValueError(
            f"gen_length={gen_length} is not divisible by "
            f"block_length={block_length}"
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError(
            f"steps={steps} is not divisible by num_blocks={num_blocks}"
        )
    steps_per_block = steps // num_blocks
    per_block = _transfer_schedule(block_length, steps_per_block)
    return per_block * num_blocks, num_blocks, steps_per_block


#: LLaDA's EOS / EoT ids, as hard-coded in the official generate.py.
LLADA_EOS_ID = 126081
LLADA_EOT_ID = 126348


def _select_predictions(
    logits: torch.Tensor,
    logits_eos_inf: bool,
    confidence_eos_eot_inf: bool,
    eos_id: int = LLADA_EOS_ID,
    eot_id: int = LLADA_EOT_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictions and confidences, matching LLaDA's generate.py at temperature 0.

    Order matters and is not obvious.  ``logits_eos_inf`` is applied *before*
    the argmax, so EOS can never be predicted.  ``confidence_eos_eot_inf`` is
    applied *after* it, so a position may still predict EOS or EoT but its
    confidence collapses and it is never chosen by the top-k.  At temperature 0
    the official ``add_gumbel_noise`` returns its argument unchanged, so the
    suppression it writes into ``logits_with_noise`` lands on ``logits`` too and
    reaches the softmax; this reproduces that.
    """
    if logits_eos_inf:
        logits[:, eos_id] = float("-inf")
    predictions = logits.argmax(dim=-1)
    if confidence_eos_eot_inf:
        logits[:, eos_id] = float("-inf")
        logits[:, eot_id] = float("-inf")
    chosen = logits.gather(1, predictions.unsqueeze(1)).squeeze(1)
    confidence = torch.exp(chosen - torch.logsumexp(logits, dim=-1))
    return predictions, confidence


@torch.no_grad()
def llada_block_rollout_states(
    model,
    prompts: torch.Tensor,
    timesteps: torch.Tensor,
    mask_id: int,
    gen_length: int,
    block_length: int,
    steps: int,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    """One true pre-forward state per prompt, under the deployed block sampler.

    ``block_length`` must be the deployed value.  Treating the whole
    generation region as a single block is what collapses an SFT'd Instruct
    checkpoint into EOS padding: LLaDA's own evaluation guide introduces block
    diffusion precisely to suppress that tendency (GSM8K 69.4 -> 78.6 at
    ``gen_length=256, block_length=8``).
    """
    if prompts.ndim != 2:
        raise ValueError("prompts must have shape [samples, tokens]")
    if bool(prompts.eq(mask_id).any()):
        raise ValueError("prompts must not already contain MASK")
    transfers, num_blocks, steps_per_block = block_transfer_schedule(
        gen_length, block_length, steps
    )
    calls = rollout_call_indices_for_transfers(timesteps, transfers)
    revealed_before_call = torch.tensor(
        [0, *torch.tensor(transfers, dtype=torch.long).cumsum(0)[:-1].tolist()]
    )
    realized = (gen_length - revealed_before_call[calls]).double() / gen_length

    device = _model_device(model)
    prompt_length = prompts.shape[1]
    collected = []
    projector = _SelectedHeadProjector(model)
    try:
        for row, target_call in enumerate(calls.tolist()):
            state = torch.full(
                (1, prompt_length + gen_length), mask_id,
                dtype=torch.long, device=device,
            )
            state[:, :prompt_length] = prompts[row : row + 1].to(device)
            done = False
            for block in range(num_blocks):
                block_end = prompt_length + (block + 1) * block_length
                for step in range(steps_per_block):
                    if block * steps_per_block + step == target_call:
                        done = True
                        break
                    selection = state.eq(mask_id)
                    # Only the current block may be revealed.
                    selection[:, block_end:] = False
                    positions = torch.nonzero(
                        selection[0], as_tuple=False
                    ).squeeze(1)
                    if positions.numel() == 0:
                        continue
                    logits = projector(state, selection)
                    predictions, confidence = _select_predictions(
                        logits, logits_eos_inf, confidence_eos_eot_inf
                    )
                    take = min(transfers[block * steps_per_block + step],
                               int(positions.numel()))
                    if take:
                        chosen = torch.topk(confidence, k=take).indices
                        state[0, positions[chosen]] = predictions[chosen]
                if done:
                    break
            collected.append(state[0].cpu().clone())
    finally:
        projector.close()

    return torch.stack(collected), {
        "rollout_call_indices": calls.tolist(),
        "rollout_transfer_schedule": transfers,
        "rollout_realized_suffix_mask_ratios": realized.tolist(),
        "rollout_gen_length": gen_length,
        "rollout_block_length": block_length,
        "rollout_num_blocks": num_blocks,
        "rollout_steps_per_block": steps_per_block,
        "rollout_logits_eos_inf": logits_eos_inf,
        "rollout_confidence_eos_eot_inf": confidence_eos_eot_inf,
        "sampler": (
            f"llada temperature=0 cfg=0 low_confidence block_diffusion "
            f"gen_length={gen_length} block_length={block_length} steps={steps} "
            f"logits_eos_inf={logits_eos_inf} "
            f"confidence_eos_eot_inf={confidence_eos_eot_inf}"
        ),
    }


def decode_revealed_samples(
    states: torch.Tensor,
    tokenizer,
    mask_id: int,
    prefix_length: int,
    count: int = 5,
) -> list[dict[str, object]]:
    """Human-readable slices of what the sampler actually wrote.

    Threshold checks catch the failure modes we already know about; reading a
    few samples is what catches the ones we do not.
    """
    suffix = states[:, prefix_length:]
    revealed = suffix.ne(mask_id)
    order = torch.argsort(revealed.sum(dim=1), descending=True)
    picks = order[:: max(1, len(order) // max(count, 1))][:count]
    samples = []
    for row in picks.tolist():
        tokens = suffix[row][revealed[row]]
        samples.append({
            "row": row,
            "revealed": int(tokens.numel()),
            "of": int(suffix.shape[1]),
            "prompt_tail": tokenizer.decode(
                states[row, max(prefix_length - 24, 0) : prefix_length].tolist()
            ),
            "generated": tokenizer.decode(tokens[:96].tolist()),
        })
    return samples


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

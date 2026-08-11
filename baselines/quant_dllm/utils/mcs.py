"""Masked Calibration Simulation shared by LLaDA and Dream."""

import torch


def timestep_for_sample(sample_index: int, num_samples: int) -> float:
    """Return the sample's point on the uniform grid (1/T, ..., T/T)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 <= sample_index < num_samples:
        raise ValueError("sample_index must be in [0, num_samples)")
    return (sample_index + 1) / num_samples


def apply_mcs(
    batch: torch.Tensor,
    mask_id: int,
    sample_index: int,
    num_samples: int,
    prefix_ratio: float = 0.25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask a calibration sample using the paper's uniform timestep coverage.

    The first ``floor(prefix_ratio * sequence_length)`` tokens remain visible.
    For timestep ``t``, every suffix token is independently masked with
    probability ``t`` (equivalently, linear visibility schedule alpha(t)=1-t).
    """
    if batch.ndim != 2:
        raise ValueError("batch must have shape [batch_size, sequence_length]")
    if batch.shape[0] != 1:
        raise ValueError("MCS currently expects calibration batch size 1")
    if not 0.0 <= prefix_ratio < 1.0:
        raise ValueError("prefix_ratio must be in [0.0, 1.0)")

    _, sequence_length = batch.shape
    prefix_length = int(prefix_ratio * sequence_length)
    timestep = timestep_for_sample(sample_index, num_samples)

    generator = torch.Generator(device=batch.device)
    generator.manual_seed(seed + sample_index)
    random_values = torch.rand(
        batch.shape,
        device=batch.device,
        generator=generator,
    )

    mask = random_values < timestep
    mask[:, :prefix_length] = False
    noisy_batch = torch.where(mask, mask_id, batch)

    mask_probability = torch.full(
        batch.shape,
        timestep,
        device=batch.device,
        dtype=torch.float32,
    )
    mask_probability[:, :prefix_length] = 0.0
    return noisy_batch, mask_probability

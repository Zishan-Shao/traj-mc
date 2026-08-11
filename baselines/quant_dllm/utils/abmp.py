"""Adaptive Blockwise Mixed Precision allocation helpers."""

from collections.abc import Sequence


def allocate_block_orders(
    salience: Sequence[float],
    base_order: int,
    ratio: float,
) -> list[int]:
    """Assign symmetric low/high orders while preserving the average order."""
    if not 0.0 <= ratio <= 0.5:
        raise ValueError("ratio must be between 0.0 and 0.5")
    if base_order < 2:
        raise ValueError("base_order must be at least 2")

    num_blocks = len(salience)
    orders = [base_order] * num_blocks
    reallocated = min(int(num_blocks * ratio), num_blocks // 2)
    if reallocated == 0:
        return orders

    ranked = sorted(range(num_blocks), key=salience.__getitem__)
    for index in ranked[:reallocated]:
        orders[index] = base_order - 1
    for index in ranked[-reallocated:]:
        orders[index] = base_order + 1
    return orders

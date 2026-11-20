"""Offline ROM construction for static quantized weight groups."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def build_lut_tables(
    weight_groups: Sequence[Sequence[int]], digit_bits: int, signed_activation: bool = False
) -> list[list[int]]:
    """Build dot-product tables for static weight groups.

    Each group yields one table. An index packs one ``digit_bits`` activation
    digit per weight, least-significant group element first.
    """
    if digit_bits <= 0:
        raise ValueError("digit_bits must be positive")
    if not weight_groups or any(not group for group in weight_groups):
        raise ValueError("weight_groups must contain non-empty groups")
    group_size = len(weight_groups[0])
    if any(len(group) != group_size for group in weight_groups):
        raise ValueError("all weight groups must have the same size")

    entries = 1 << (group_size * digit_bits)
    return [
        [_dot_for_index(group, index, digit_bits, signed_activation) for index in range(entries)]
        for group in weight_groups
    ]


def write_lut_mem(path: str | Path, tables: Sequence[Sequence[int]], output_bits: int) -> Path:
    """Write tables in $readmemh-compatible two's-complement hexadecimal."""
    if output_bits <= 0:
        raise ValueError("output_bits must be positive")
    if not tables or any(not table for table in tables):
        raise ValueError("tables must contain at least one non-empty table")
    width = (output_bits + 3) // 4
    mask = (1 << output_bits) - 1
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(f"{value & mask:0{width}x}" for table in tables for value in table) + "\n",
        encoding="ascii",
    )
    return target


def _dot_for_index(
    weights: Sequence[int], index: int, digit_bits: int, signed_activation: bool
) -> int:
    mask = (1 << digit_bits) - 1
    values = []
    for offset in range(len(weights)):
        value = (index >> (offset * digit_bits)) & mask
        if signed_activation and value & (1 << (digit_bits - 1)):
            value -= 1 << digit_bits
        values.append(value)
    return sum(value * weight for value, weight in zip(values, weights))

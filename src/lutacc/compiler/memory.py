"""Lifetime-aware SRAM planning and Omni-LUT KV tiling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..schema import AttentionShape
from .ir import Program, TensorSpec


DTYPE_BITS = {
    "fp16": 16,
    "fp32": 32,
    "int2": 2,
    "int3": 3,
    "int4": 4,
    "uint4": 4,
    "uint8": 8,
    "index3": 3,
    "index4": 4,
}


def tensor_bytes(tensor: TensorSpec) -> int:
    elements = math.prod(tensor.shape)
    return int(math.ceil(elements * DTYPE_BITS[tensor.dtype] / 8.0))


@dataclass(frozen=True)
class MemoryAllocation:
    tensor: str
    offset: Optional[int]
    size_bytes: int
    first_operation: int
    last_operation: int
    spilled: bool


@dataclass(frozen=True)
class MemoryPlan:
    capacity_bytes: int
    peak_bytes: int
    allocations: Tuple[MemoryAllocation, ...]

    @property
    def spill_bytes(self) -> int:
        return sum(item.size_bytes for item in self.allocations if item.spilled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "peak_bytes": self.peak_bytes,
            "spill_bytes": self.spill_bytes,
            "allocations": [asdict(item) for item in self.allocations],
        }


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def plan_sram(program: Program, capacity_bytes: int, alignment: int = 64) -> MemoryPlan:
    if capacity_bytes <= 0 or alignment <= 0:
        raise ValueError("capacity and alignment must be positive")
    program.validate()
    operations = program.topological_operations()
    operation_index = {operation.name: index for index, operation in enumerate(operations)}
    producer: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for index, operation in enumerate(operations):
        for output in operation.outputs:
            producer[output] = index
        for input_name in operation.inputs:
            consumers.setdefault(input_name, []).append(index)
    lifetimes = []
    for tensor in program.tensors:
        if tensor.storage != "sram":
            continue
        first = producer.get(tensor.name, 0)
        last = max(consumers.get(tensor.name, [first]))
        lifetimes.append((first, last, tensor.name, _align(tensor_bytes(tensor), alignment)))
    lifetimes.sort(key=lambda item: (item[0], -item[3], item[2]))

    active: List[MemoryAllocation] = []
    allocations: List[MemoryAllocation] = []
    peak = 0
    for first, last, name, size in lifetimes:
        active = [item for item in active if item.last_operation >= first and not item.spilled]
        occupied = sorted(
            (item.offset, item.offset + item.size_bytes)
            for item in active
            if item.offset is not None
        )
        offset = 0
        for begin, end in occupied:
            if offset + size <= begin:
                break
            offset = max(offset, end)
        spilled = offset + size > capacity_bytes
        allocation = MemoryAllocation(
            name,
            None if spilled else offset,
            size,
            first,
            last,
            spilled,
        )
        allocations.append(allocation)
        if not spilled:
            active.append(allocation)
            peak = max(peak, offset + size)
    return MemoryPlan(capacity_bytes, peak, tuple(allocations))


def plan_omni_kv_tiles(
    shape: AttentionShape,
    key_bits: float,
    value_bits: float,
    capacity_bytes: int,
) -> Dict[str, Any]:
    if not 2 <= key_bits <= 4 or not 2 <= value_bits <= 4:
        raise ValueError("KV precisions must be in [2, 4]")
    if capacity_bytes <= 0:
        raise ValueError("KV SRAM capacity must be positive")
    bytes_per_token = int(
        math.ceil(
            shape.batch
            * shape.heads
            * shape.head_dim
            * (key_bits + value_bits)
            / 8.0
        )
    )
    resident_tokens = max(1, capacity_bytes // bytes_per_token)
    tile_tokens = min(shape.kv_length, resident_tokens)
    tiles = int(math.ceil(shape.kv_length / tile_tokens))
    return {
        "kv_length": shape.kv_length,
        "bytes_per_token": bytes_per_token,
        "total_payload_bytes": bytes_per_token * shape.kv_length,
        "capacity_bytes": capacity_bytes,
        "resident_tokens": resident_tokens,
        "tile_tokens": tile_tokens,
        "tiles": tiles,
        "last_tile_tokens": shape.kv_length - tile_tokens * (tiles - 1),
        "requires_dram_tiling": tiles > 1,
    }

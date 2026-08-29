"""Cycle-level open surrogate for Omni-LUT attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import default_hardware_path, load_yaml, require_positive
from ..schema import AttentionShape, GemmShape, SimulationResult
from .event import EventScheduler, EventSpec, makespan


def _ceildiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class OmniCycleModel:
    name: str
    clock_hz: float
    array_rows: int
    array_cols: int
    lut_pes: int
    lut_group_size: int
    lookups_per_pe_cycle: int
    bqu_values_per_cycle: int
    lgu_entries_per_cycle: int
    softmax_values_per_cycle: int
    kv_sram_bytes: int
    lut_sram_bytes: int
    dram_bandwidth_bytes_per_cycle: int
    idle_power_w: float
    active_power_w: float

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "OmniCycleModel":
        config = load_yaml(str(path or default_hardware_path("omni")))
        keys = (
            "clock_hz",
            "array_rows",
            "array_cols",
            "lut_pes",
            "lut_group_size",
            "lookups_per_pe_cycle",
            "bqu_values_per_cycle",
            "lgu_entries_per_cycle",
            "softmax_values_per_cycle",
            "kv_sram_bytes",
            "lut_sram_bytes",
            "dram_bandwidth_bytes_per_cycle",
            "idle_power_w",
            "active_power_w",
        )
        require_positive(config, *keys)
        values = {key: config[key] for key in ("name",) + keys}
        model = cls(**values)
        if model.lut_pes != model.array_rows * model.array_cols:
            raise ValueError("Omni-LUT PE count must equal array_rows*array_cols")
        return model

    @property
    def lookup_capacity(self) -> int:
        return self.lut_pes * self.lookups_per_pe_cycle


def simulate_omni_gemm(
    shape: GemmShape,
    weight_bits: int = 4,
    model: Optional[OmniCycleModel] = None,
    resident_weights: bool = True,
) -> SimulationResult:
    """Inferred cycle model for Omni-LUT activation-weight GEMM."""

    if weight_bits != 4:
        raise ValueError("the open Omni-LUT AW cycle model supports W4 weights")
    cycle_model = model or OmniCycleModel.from_file()
    tokens = shape.batch * shape.m
    levels = 1 << weight_bits
    lgu_entries = tokens * shape.k * levels
    lgu_cycles = _ceildiv(lgu_entries, cycle_model.lgu_entries_per_cycle)
    lookups = shape.macs
    output_tiles = _ceildiv(shape.n, cycle_model.array_cols)
    token_tiles = _ceildiv(tokens, cycle_model.array_rows)
    array_fill = token_tiles * output_tiles * (
        cycle_model.array_rows + cycle_model.array_cols - 2
    )
    array_cycles = _ceildiv(lookups, cycle_model.lookup_capacity) + array_fill
    activation_bytes = tokens * shape.k * 2
    packed_weight_bytes = _ceildiv(shape.k * shape.n * weight_bits, 8)
    output_bytes = tokens * shape.n * 2
    events = [
        EventSpec(
            "lgu_aw",
            "lgu",
            lgu_cycles,
            operations=lgu_entries,
            read_bytes=activation_bytes + levels * 2,
            write_bytes=lgu_entries * 2,
            metadata={"table_scope": "activation_x_weight_centroid"},
        ),
        EventSpec(
            "aw_lut_array",
            "lut_pe_array",
            array_cycles,
            ("lgu_aw",),
            operations=2 * shape.macs,
            read_bytes=packed_weight_bytes + lgu_entries * 2,
            write_bytes=output_bytes,
            metadata={"dataflow": "weight_stationary"},
        ),
    ]
    stages = EventScheduler().schedule(events)
    core_cycles = makespan(stages)
    dram_read = activation_bytes + (0 if resident_weights else packed_weight_bytes)
    dram_write = output_bytes
    dram_cycles = _ceildiv(
        dram_read + dram_write, cycle_model.dram_bandwidth_bytes_per_cycle
    )
    total_cycles = max(core_cycles, dram_cycles)
    activity = min(
        1.0, sum(stage.busy_cycles for stage in stages) / (total_cycles * 2.0)
    )
    average_power = cycle_model.idle_power_w + (
        cycle_model.active_power_w - cycle_model.idle_power_w
    ) * activity
    return SimulationResult(
        design="omni-lut-aw",
        workload="gemm-%dx%dx%d-b%d" % (shape.m, shape.k, shape.n, shape.batch),
        clock_hz=cycle_model.clock_hz,
        cycles=total_cycles,
        stages=stages,
        operations=2 * shape.macs,
        dram_read_bytes=dram_read,
        dram_write_bytes=dram_write,
        sram_read_bytes=sum(stage.read_bytes for stage in stages),
        sram_write_bytes=sum(stage.write_bytes for stage in stages),
        energy_mj=average_power * total_cycles / cycle_model.clock_hz * 1000.0,
        metadata={
            "core_cycles": core_cycles,
            "dram_cycles": dram_cycles,
            "dataflow": "weight_stationary",
            "weight_bits": weight_bits,
            "lgu_entries": lgu_entries,
            "lut_lookups": lookups,
            "resident_weights": resident_weights,
            "configuration_evidence": "inferred",
        },
    )


def simulate_omni_attention(
    shape: AttentionShape,
    phase: str,
    average_key_bits: float = 3.0,
    average_value_bits: float = 3.0,
    model: Optional[OmniCycleModel] = None,
) -> SimulationResult:
    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be prefill or decode")
    if not 2 <= average_key_bits <= 4 or not 2 <= average_value_bits <= 4:
        raise ValueError("AS-Bit averages must be in [2, 4]")
    cycle_model = model or OmniCycleModel.from_file()
    groups = shape.batch * shape.heads
    new_kv_tokens = shape.kv_length if phase == "prefill" else 1
    bqu_values = groups * new_kv_tokens * shape.head_dim * 2
    bqu_cycles = _ceildiv(bqu_values, cycle_model.bqu_values_per_cycle)

    key_levels = 1 << int(math.ceil(average_key_bits))
    value_levels = 1 << int(math.ceil(average_value_bits))
    channel_groups = _ceildiv(shape.head_dim, cycle_model.lut_group_size)
    qk_lgu_entries = groups * shape.query_length * channel_groups * key_levels
    pv_lgu_entries = groups * shape.query_length * shape.kv_length * value_levels
    qk_lgu_cycles = _ceildiv(qk_lgu_entries, cycle_model.lgu_entries_per_cycle)
    pv_lgu_cycles = _ceildiv(pv_lgu_entries, cycle_model.lgu_entries_per_cycle)

    qk_lookups = _ceildiv(shape.qk_macs, cycle_model.lut_group_size)
    pv_lookups = _ceildiv(shape.pv_macs, cycle_model.lut_group_size)
    qk_base = _ceildiv(qk_lookups, cycle_model.lookup_capacity)
    pv_base = _ceildiv(pv_lookups, cycle_model.lookup_capacity)
    query_tiles = _ceildiv(shape.query_length, cycle_model.array_rows)
    output_tiles = _ceildiv(shape.head_dim, cycle_model.array_cols)
    array_fill = query_tiles * output_tiles * (cycle_model.array_rows + cycle_model.array_cols - 2)
    if phase == "prefill":
        qk_cycles = qk_base + array_fill
        pv_cycles = pv_base + array_fill
        dataflow = "prefill_ws"
        dataflow_efficiency = 1.0
    else:
        # OS-V keeps partial outputs and value tiles resident during one-token
        # decoding. 0.85 is an explicitly inferred issue-efficiency parameter,
        # not a paper result bar.
        qk_cycles = int(math.ceil(qk_base / 0.85)) + array_fill
        pv_cycles = int(math.ceil(pv_base / 0.90)) + array_fill
        dataflow = "decode_os_v"
        dataflow_efficiency = 0.875
    score_values = groups * shape.query_length * shape.kv_length
    softmax_cycles = _ceildiv(score_values, cycle_model.softmax_values_per_cycle)

    key_payload = int(
        math.ceil(groups * shape.kv_length * shape.head_dim * average_key_bits / 8.0)
    )
    value_payload = int(
        math.ceil(groups * shape.kv_length * shape.head_dim * average_value_bits / 8.0)
    )
    query_bytes = groups * shape.query_length * shape.head_dim * 2
    output_bytes = query_bytes
    total_kv_payload = key_payload + value_payload
    kv_tile_count = max(1, _ceildiv(total_kv_payload, cycle_model.kv_sram_bytes))
    kv_streamed_from_dram = phase == "prefill" or kv_tile_count > 1
    events = [
        EventSpec(
            "bqu_kv",
            "bqu",
            bqu_cycles,
            operations=bqu_values,
            read_bytes=bqu_values * 2,
            write_bytes=int(
                math.ceil(
                    groups
                    * new_kv_tokens
                    * shape.head_dim
                    * (average_key_bits + average_value_bits)
                    / 8.0
                )
            ),
        ),
        EventSpec(
            "lgu_qk",
            "lgu",
            qk_lgu_cycles,
            ("bqu_kv",),
            operations=qk_lgu_entries,
        ),
        EventSpec(
            "qk_lut_array",
            "lut_pe_array",
            qk_cycles,
            ("lgu_qk",),
            operations=2 * shape.qk_macs,
            read_bytes=query_bytes + key_payload,
            write_bytes=score_values * 2,
            metadata={"dataflow": dataflow},
        ),
        EventSpec(
            "softmax",
            "special_function",
            softmax_cycles,
            ("qk_lut_array",),
            operations=5 * score_values,
            read_bytes=score_values * 2,
            write_bytes=score_values * 2,
        ),
        EventSpec(
            "lgu_pv",
            "lgu",
            pv_lgu_cycles,
            ("softmax",),
            operations=pv_lgu_entries,
        ),
        EventSpec(
            "pv_lut_array",
            "lut_pe_array",
            pv_cycles,
            ("lgu_pv",),
            operations=2 * shape.pv_macs,
            read_bytes=score_values * 2 + value_payload,
            write_bytes=output_bytes,
            metadata={"dataflow": dataflow},
        ),
    ]
    stages = EventScheduler().schedule(events)
    core_cycles = makespan(stages)
    # Decode can reuse the cache only while its full packed payload fits the
    # modeled KV SRAM.  Beyond that point every query streams all tiles once;
    # hiding this transfer would make long-context system traffic constant.
    dram_read = query_bytes + (total_kv_payload if kv_streamed_from_dram else 0)
    dram_write = output_bytes + int(
        math.ceil(
            groups
            * new_kv_tokens
            * shape.head_dim
            * (average_key_bits + average_value_bits)
            / 8.0
        )
    )
    dram_cycles = _ceildiv(
        dram_read + dram_write, cycle_model.dram_bandwidth_bytes_per_cycle
    )
    total_cycles = max(core_cycles, dram_cycles)
    activity = min(1.0, sum(stage.busy_cycles for stage in stages) / (total_cycles * 4.0))
    average_power = cycle_model.idle_power_w + (
        cycle_model.active_power_w - cycle_model.idle_power_w
    ) * activity
    energy_mj = average_power * total_cycles / cycle_model.clock_hz * 1000.0
    sram_reads = sum(stage.read_bytes for stage in stages)
    sram_writes = sum(stage.write_bytes for stage in stages)
    return SimulationResult(
        design="omni-lut",
        workload="attention-%s-q%d-kv%d-d%d-b%d-h%d"
        % (
            phase,
            shape.query_length,
            shape.kv_length,
            shape.head_dim,
            shape.batch,
            shape.heads,
        ),
        clock_hz=cycle_model.clock_hz,
        cycles=total_cycles,
        stages=stages,
        operations=2 * (shape.qk_macs + shape.pv_macs) + 5 * score_values,
        dram_read_bytes=dram_read,
        dram_write_bytes=dram_write,
        sram_read_bytes=sram_reads,
        sram_write_bytes=sram_writes,
        energy_mj=energy_mj,
        metadata={
            "core_cycles": core_cycles,
            "dram_cycles": dram_cycles,
            "dataflow": dataflow,
            "dataflow_issue_efficiency": dataflow_efficiency,
            "array_shape": [cycle_model.array_rows, cycle_model.array_cols],
            "key_cache_bytes": key_payload,
            "value_cache_bytes": value_payload,
            "average_key_bits": average_key_bits,
            "average_value_bits": average_value_bits,
            "kv_tile_count": kv_tile_count,
            "kv_streamed_from_dram": kv_streamed_from_dram,
            "configuration_evidence": "inferred",
        },
    )

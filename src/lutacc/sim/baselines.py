"""Mechanism-based open baseline models used in paper comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import load_yaml, repository_root, require_positive
from ..schema import GemmShape, SimulationResult
from .event import EventScheduler, EventSpec, makespan


def _ceildiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class FIGLUTCycleModel:
    name: str
    clock_hz: float
    group_size: int
    weight_bits: int
    parallel_outputs: int
    group_bit_lanes_per_output: int
    dram_bandwidth_bytes_per_cycle: int
    power_w: float

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "FIGLUTCycleModel":
        source = path or repository_root() / "configs" / "hardware" / "figlut_scaled.yaml"
        config = load_yaml(str(source))
        keys = (
            "clock_hz",
            "group_size",
            "weight_bits",
            "parallel_outputs",
            "group_bit_lanes_per_output",
            "dram_bandwidth_bytes_per_cycle",
            "power_w",
        )
        require_positive(config, *keys)
        return cls(**{key: config[key] for key in ("name",) + keys})


@dataclass(frozen=True)
class GPURooflineModel:
    name: str
    timebase_hz: float
    weight_bits: int
    effective_bandwidth_bytes_per_second: float
    peak_compute_ops_per_second: float
    active_power_w: float
    evidence: str = "calibrated"

    @classmethod
    def from_file(
        cls, baseline: str, path: Optional[str] = None
    ) -> "GPURooflineModel":
        source = path or (
            repository_root()
            / "configs"
            / "calibration"
            / "gpu_decode_roofline_v1.yaml"
        )
        config = load_yaml(str(source))
        if baseline not in {"a100_fp16", "quarot_w4a4"}:
            raise ValueError("GPU baseline must be a100_fp16 or quarot_w4a4")
        local = config[baseline]
        require_positive(
            local,
            "weight_bits",
            "effective_bandwidth_bytes_per_second",
            "peak_compute_ops_per_second",
            "active_power_w",
        )
        return cls(
            name=baseline,
            timebase_hz=float(config["timebase_hz"]),
            weight_bits=int(local["weight_bits"]),
            effective_bandwidth_bytes_per_second=float(
                local["effective_bandwidth_bytes_per_second"]
            ),
            peak_compute_ops_per_second=float(local["peak_compute_ops_per_second"]),
            active_power_w=float(local["active_power_w"]),
            evidence=str(config["evidence"]),
        )


def simulate_gpu_roofline_gemm(
    shape: GemmShape,
    baseline: str,
    model: Optional[GPURooflineModel] = None,
) -> SimulationResult:
    """Single-batch GEMV/GEMM roofline used for open GPU comparisons."""

    cycle_model = model or GPURooflineModel.from_file(baseline)
    tokens = shape.batch * shape.m
    activation_bytes = tokens * shape.k * 2
    weight_bytes = _ceildiv(shape.k * shape.n * cycle_model.weight_bits, 8)
    output_bytes = tokens * shape.n * 2
    total_bytes = activation_bytes + weight_bytes + output_bytes
    operations = 2 * shape.macs
    memory_seconds = total_bytes / cycle_model.effective_bandwidth_bytes_per_second
    compute_seconds = operations / cycle_model.peak_compute_ops_per_second
    bound_seconds = max(memory_seconds, compute_seconds)
    total_cycles = max(1, int(math.ceil(bound_seconds * cycle_model.timebase_hz)))
    bound = "memory" if memory_seconds >= compute_seconds else "compute"
    stages = EventScheduler().schedule(
        [
            EventSpec(
                "gpu_roofline_%s" % bound,
                "gpu_roofline",
                total_cycles,
                operations=operations,
                read_bytes=activation_bytes + weight_bytes,
                write_bytes=output_bytes,
                metadata={
                    "memory_seconds": memory_seconds,
                    "compute_seconds": compute_seconds,
                },
            )
        ]
    )
    return SimulationResult(
        design=baseline.replace("_", "-"),
        workload="gemm-%dx%dx%d-b%d" % (shape.m, shape.k, shape.n, shape.batch),
        clock_hz=cycle_model.timebase_hz,
        cycles=total_cycles,
        stages=stages,
        operations=operations,
        dram_read_bytes=activation_bytes + weight_bytes,
        dram_write_bytes=output_bytes,
        sram_read_bytes=0,
        sram_write_bytes=0,
        energy_mj=cycle_model.active_power_w * bound_seconds * 1000.0,
        metadata={
            "roofline_bound": bound,
            "weight_bits": cycle_model.weight_bits,
            "effective_bandwidth_bytes_per_second": (
                cycle_model.effective_bandwidth_bytes_per_second
            ),
            "peak_compute_ops_per_second": cycle_model.peak_compute_ops_per_second,
            "active_power_w": cycle_model.active_power_w,
            "configuration_evidence": "calibrated_roofline_not_native_measurement",
        },
    )


def simulate_figlut_gemm(
    shape: GemmShape,
    model: Optional[FIGLUTCycleModel] = None,
    resident_weights: bool = False,
) -> SimulationResult:
    cycle_model = model or FIGLUTCycleModel.from_file()
    tokens = shape.batch * shape.m
    output_waves = _ceildiv(tokens * shape.n, cycle_model.parallel_outputs)
    reduction_groups = _ceildiv(shape.k, cycle_model.group_size)
    bit_serial_work = reduction_groups * cycle_model.weight_bits
    query_cycles = output_waves * _ceildiv(
        bit_serial_work, cycle_model.group_bit_lanes_per_output
    )
    # LUT construction is activation-dependent. The half-FFLUT generator builds
    # one group table while the prior group is queried, so only fill/drain stays
    # exposed at the beginning/end of a GEMM.
    lut_generation_fill = _ceildiv(
        tokens * shape.k,
        cycle_model.parallel_outputs * cycle_model.group_size,
    )
    compute_cycles = query_cycles + lut_generation_fill
    activation_bytes = tokens * shape.k * 2
    weight_bytes = _ceildiv(shape.k * shape.n * cycle_model.weight_bits, 8)
    output_bytes = tokens * shape.n * 2
    dram_read = activation_bytes + (0 if resident_weights else weight_bytes)
    dram_cycles = _ceildiv(
        dram_read + output_bytes, cycle_model.dram_bandwidth_bytes_per_cycle
    )
    stages = EventScheduler().schedule(
        [
            EventSpec(
                "lut_generation_fill",
                "lut_generator",
                lut_generation_fill,
                operations=tokens * shape.k,
                read_bytes=activation_bytes,
            ),
            EventSpec(
                "bit_serial_lut_query",
                "rac_array",
                query_cycles,
                ("lut_generation_fill",),
                operations=2 * shape.macs,
                read_bytes=weight_bytes,
                write_bytes=output_bytes,
            ),
        ]
    )
    core_cycles = makespan(stages)
    total_cycles = max(core_cycles, dram_cycles)
    energy_mj = cycle_model.power_w * total_cycles / cycle_model.clock_hz * 1000.0
    return SimulationResult(
        design="figlut-q4",
        workload="gemm-%dx%dx%d-b%d" % (shape.m, shape.k, shape.n, shape.batch),
        clock_hz=cycle_model.clock_hz,
        cycles=total_cycles,
        stages=stages,
        operations=2 * shape.macs,
        dram_read_bytes=dram_read,
        dram_write_bytes=output_bytes,
        sram_read_bytes=weight_bytes + output_bytes,
        sram_write_bytes=output_bytes,
        energy_mj=energy_mj,
        metadata={
            "core_cycles": core_cycles,
            "dram_cycles": dram_cycles,
            "group_size": cycle_model.group_size,
            "bit_serial_planes": cycle_model.weight_bits,
            "output_waves": output_waves,
            "configuration_evidence": "inferred_from_figlut_primary_source",
        },
    )

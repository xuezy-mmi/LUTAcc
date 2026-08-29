"""Mechanism-level OASIS cycle, traffic, and energy model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..config import default_hardware_path, load_yaml, repository_root, require_positive
from ..schema import GemmShape, SimulationResult
from .event import EventScheduler, EventSpec, makespan


def _ceildiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class OasisCycleModel:
    name: str
    clock_hz: float
    pe_lines: int
    concat_units_per_line: int
    index_counters_per_line: int
    index_counter_inputs: int
    index_counter_dual_issue: int
    mac_tree_inputs: int
    compensation_macs_per_line: int
    clustering_units: int
    orizuru_units: int
    orizuru_inputs: int
    power_w: float
    dram_bandwidth_bytes_per_cycle: int
    activation_values_per_cycle: int = 2
    index_values_per_wave_cycle: int = 1024
    reduction_pipeline_overhead_cycles: int = 1
    orizuru_local_init_cycles: int = 3
    orizuru_global_fill_per_level: int = 8
    dequant_pipeline_overhead_cycles: int = 1
    merge_values_per_line_cycle: int = 1

    @classmethod
    def from_files(
        cls,
        hardware_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
    ) -> "OasisCycleModel":
        hardware = load_yaml(str(hardware_path or default_hardware_path("oasis")))
        calibration = load_yaml(
            str(
                calibration_path
                or repository_root() / "configs" / "calibration" / "oasis_pipeline_v1.yaml"
            )
        )
        require_positive(
            hardware,
            "clock_hz",
            "pe_lines",
            "concat_units_per_line",
            "index_counters_per_line",
            "index_counter_inputs",
            "index_counter_dual_issue",
            "mac_tree_inputs",
            "compensation_macs_per_line",
            "clustering_units",
            "orizuru_units",
            "orizuru_inputs",
            "power_w",
            "dram_bandwidth_bytes_per_cycle",
        )
        require_positive(
            calibration,
            "activation_values_per_cycle",
            "index_values_per_wave_cycle",
            "reduction_pipeline_overhead_cycles",
            "orizuru_local_init_cycles",
            "orizuru_global_fill_per_level",
            "dequant_pipeline_overhead_cycles",
            "merge_values_per_line_cycle",
        )
        fields = {
            key: hardware[key]
            for key in (
                "name",
                "clock_hz",
                "pe_lines",
                "concat_units_per_line",
                "index_counters_per_line",
                "index_counter_inputs",
                "index_counter_dual_issue",
                "mac_tree_inputs",
                "compensation_macs_per_line",
                "clustering_units",
                "orizuru_units",
                "orizuru_inputs",
                "power_w",
                "dram_bandwidth_bytes_per_cycle",
            )
        }
        fields.update(
            {
                key: calibration[key]
                for key in (
                    "activation_values_per_cycle",
                    "index_values_per_wave_cycle",
                    "reduction_pipeline_overhead_cycles",
                    "orizuru_local_init_cycles",
                    "orizuru_global_fill_per_level",
                    "dequant_pipeline_overhead_cycles",
                    "merge_values_per_line_cycle",
                )
            }
        )
        return cls(**fields)

    def cycle_ledger(
        self,
        shape: GemmShape,
        activation_bits: int,
        outlier_fraction: float,
    ) -> Dict[str, int]:
        if activation_bits not in (3, 4):
            raise ValueError("OASIS activation precision must be 3 or 4 bits")
        if not 0 <= outlier_fraction < 1:
            raise ValueError("outlier_fraction must be in [0, 1)")
        tokens = shape.batch * shape.m
        output_waves = _ceildiv(tokens * shape.n, self.pe_lines)
        reduction_tiles = _ceildiv(shape.k, self.concat_units_per_line)
        activation_quantization = _ceildiv(
            tokens * shape.k, self.activation_values_per_cycle
        )
        concat = output_waves * reduction_tiles
        index_distribution = output_waves * _ceildiv(
            shape.k, self.index_values_per_wave_cycle
        )
        lut_entries = 1 << (4 + activation_bits)
        reduction = output_waves * (
            _ceildiv(lut_entries, self.mac_tree_inputs)
            + self.reduction_pipeline_overhead_cycles
        )

        k_each = int(shape.k * outlier_fraction * 0.5)
        tree_levels = max(1, int(math.ceil(math.log(shape.k, 2))))
        units_per_token = _ceildiv(shape.k, self.orizuru_inputs)
        token_parallelism = max(1, self.orizuru_units // units_per_token)
        detector_token_waves = _ceildiv(tokens, token_parallelism)
        outlier_detection_per_wave = (
            units_per_token * self.orizuru_local_init_cycles
            + 2 * k_each * tree_levels
            + self.orizuru_global_fill_per_level * tree_levels
        )
        outlier_detection = detector_token_waves * outlier_detection_per_wave
        total_outliers = 2 * k_each
        error_compensation = output_waves * (
            _ceildiv(total_outliers, self.compensation_macs_per_line)
            + self.dequant_pipeline_overhead_cycles
        )
        merge = _ceildiv(
            tokens * shape.n,
            self.pe_lines * self.merge_values_per_line_cycle,
        )
        main = max(activation_quantization, concat, index_distribution, reduction)
        outlier = max(outlier_detection, error_compensation)
        return {
            "activation_quantization": activation_quantization,
            "concat": concat,
            "index_distribution": index_distribution,
            "reduction": reduction,
            "outlier_detection": outlier_detection,
            "error_compensation": error_compensation,
            "main_branch": main,
            "outlier_branch": outlier,
            "merge": merge,
            "total": max(main, outlier) + merge,
            "output_waves": output_waves,
            "lut_entries": lut_entries,
            "outliers_per_token": total_outliers,
        }


_ANCHOR_STAGE_CYCLES = {
    "activation_quantization": 2048,
    "concat": 256,
    "index_distribution": 1024,
    "reduction": 2304,
    "outlier_detection": 1344,
    "error_compensation": 1536,
    "merge": 256,
}
_ANCHOR_ENERGY_PERCENT = {
    "activation_quantization": 14.7,
    "concat": 1.8,
    "index_distribution": 7.4,
    "reduction": 33.1,
    "outlier_detection": 17.4,
    "error_compensation": 3.6,
    "merge": 22.1,
}


def _stage_energies(model: OasisCycleModel, ledger: Mapping[str, int]) -> Dict[str, float]:
    anchor_total_nj = model.power_w * (2560.0 / model.clock_hz) * 1e9
    normalized = sum(_ANCHOR_ENERGY_PERCENT.values())
    energies = {}
    for name, percentage in _ANCHOR_ENERGY_PERCENT.items():
        anchor_energy = anchor_total_nj * percentage / normalized
        energies[name] = anchor_energy * ledger[name] / _ANCHOR_STAGE_CYCLES[name]
    return energies


def _traffic(
    shape: GemmShape, activation_bits: int, lut_entries: int, output_waves: int
) -> Dict[str, int]:
    tokens = shape.batch * shape.m
    weight_index = _ceildiv(shape.k * shape.n * 4, 8)
    activation_index = _ceildiv(tokens * shape.k * activation_bits, 8) * output_waves
    lut = shape.n * lut_entries * 2
    output = tokens * shape.n * 2
    return {
        "weight_index_buffer": weight_index,
        "activation_index_buffer": activation_index,
        "lut": lut,
        "output_buffer": output,
    }


def simulate_oasis_gemm(
    shape: GemmShape,
    activation_bits: int = 4,
    outlier_fraction: float = 0.01,
    model: Optional[OasisCycleModel] = None,
    resident_weights: bool = True,
    lookahead: bool = True,
) -> SimulationResult:
    cycle_model = model or OasisCycleModel.from_files()
    ledger = cycle_model.cycle_ledger(shape, activation_bits, outlier_fraction)
    energy = _stage_energies(cycle_model, ledger)
    traffic = _traffic(
        shape,
        activation_bits,
        ledger["lut_entries"],
        ledger["output_waves"],
    )
    tokens = shape.batch * shape.m
    main_components = {
        key: ledger[key]
        for key in ("activation_quantization", "concat", "index_distribution", "reduction")
    }
    detector_dependencies = ()
    branch_dependencies = () if lookahead else ("outlier_detection",)
    events = [
        EventSpec(
            "outlier_detection",
            "orizuru",
            ledger["outlier_detection"],
            detector_dependencies,
            operations=tokens
            * int(1.5 * shape.k + ledger["outliers_per_token"] * math.ceil(math.log(shape.k, 2))),
            energy_nj=energy["outlier_detection"],
            metadata={"lookahead": lookahead},
        ),
        EventSpec(
            "main_branch",
            "oasis_main",
            ledger["main_branch"],
            branch_dependencies,
            operations=shape.macs,
            read_bytes=traffic["activation_index_buffer"] + traffic["lut"],
            write_bytes=tokens * shape.n * 2,
            energy_nj=sum(energy[name] for name in main_components),
            metadata={"overlapped_components": main_components},
        ),
        EventSpec(
            "error_compensation",
            "compensation_macs",
            ledger["error_compensation"],
            branch_dependencies,
            operations=tokens * shape.n * ledger["outliers_per_token"] * 2,
            read_bytes=traffic["weight_index_buffer"],
            energy_nj=energy["error_compensation"],
            metadata={"streamed_after_orizuru": True},
        ),
        EventSpec(
            "merge",
            "merge_macs",
            ledger["merge"],
            ("main_branch", "outlier_detection", "error_compensation"),
            operations=tokens * shape.n,
            read_bytes=tokens * shape.n * 4,
            write_bytes=tokens * shape.n * 2,
            energy_nj=energy["merge"],
        ),
    ]
    stages = EventScheduler().schedule(events)
    core_cycles = makespan(stages)
    activation_read = tokens * shape.k * 2
    weight_read = 0 if resident_weights else traffic["weight_index_buffer"]
    output_write = tokens * shape.n * 2
    dram_read = activation_read + weight_read
    dram_cycles = _ceildiv(dram_read + output_write, cycle_model.dram_bandwidth_bytes_per_cycle)
    total_cycles = max(core_cycles, dram_cycles)
    # The synthesis-derived stage model covers the core. Add an explicit,
    # separately visible 28-nm/HBM proxy for external transfers.
    memory_energy_nj = (dram_read + output_write) * 0.0336
    energy_mj = (sum(stage.energy_nj for stage in stages) + memory_energy_nj) / 1e6
    sram_total = sum(traffic.values())
    traffic_percent = {
        name: 100.0 * value / sram_total for name, value in traffic.items()
    }
    core_energy_total = sum(energy.values())
    stage_energy_percent = {
        name: 100.0 * value / core_energy_total for name, value in energy.items()
    }
    return SimulationResult(
        design="oasis-a%d" % activation_bits,
        workload="gemm-%dx%dx%d-b%d" % (shape.m, shape.k, shape.n, shape.batch),
        clock_hz=cycle_model.clock_hz,
        cycles=total_cycles,
        stages=stages,
        operations=2 * shape.macs,
        dram_read_bytes=dram_read,
        dram_write_bytes=output_write,
        sram_read_bytes=sram_total,
        sram_write_bytes=tokens * shape.k * activation_bits // 8 + output_write,
        energy_mj=energy_mj,
        metadata={
            "cycle_ledger": ledger,
            "core_cycles": core_cycles,
            "dram_cycles": dram_cycles,
            "resident_weights": resident_weights,
            "lookahead": lookahead,
            "sram_traffic_bytes": traffic,
            "sram_traffic_percent": traffic_percent,
            "energy_model": "oasis_arxiv_v4_28nm_stage_calibration",
            "stage_energy_nj": energy,
            "stage_energy_percent": stage_energy_percent,
            "memory_energy_nj": memory_energy_nj,
        },
    )

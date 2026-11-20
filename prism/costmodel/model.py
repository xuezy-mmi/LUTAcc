"""Analytical, calibration-friendly cost model for mixed-precision GEMMs.

The model intentionally reports relative costs (the default technology constants
are not tied to a particular process). Replace :class:`TechnologyConfig` with
measurements from synthesis or a power library before making absolute claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Primitive(str, Enum):
    MAC = "mac"
    LUT = "lut"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class PrecisionSpec:
    """Operand precision in bits.

    ``kind`` is ``int`` or ``fp``. Floating point is represented by its total
    width for early exploration; a calibrated model can refine this later.
    """

    bits: int
    kind: str = "int"
    signed: bool = True

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("precision bits must be positive")
        if self.kind not in {"int", "fp"}:
            raise ValueError("kind must be 'int' or 'fp'")


@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("GEMM dimensions must be positive")


@dataclass(frozen=True)
class ArrayConfig:
    rows: int
    cols: int
    tile_k: int = 16
    dataflow: str = "output_stationary"

    def __post_init__(self) -> None:
        if min(self.rows, self.cols, self.tile_k) <= 0:
            raise ValueError("array dimensions and tile_k must be positive")
        if self.dataflow not in {
            "output_stationary",
            "weight_stationary",
            "input_stationary",
        }:
            raise ValueError(
                "dataflow must be output_stationary, weight_stationary, or input_stationary"
            )


@dataclass(frozen=True)
class LutConfig:
    group_size: int = 1
    digit_bits: int = 4
    sharing: str = "per_pe"
    max_index_bits: int = 16

    def __post_init__(self) -> None:
        if self.group_size <= 0 or self.digit_bits <= 0:
            raise ValueError("LUT group_size and digit_bits must be positive")
        if self.sharing not in {"per_pe", "per_tile", "global"}:
            raise ValueError("sharing must be per_pe, per_tile, or global")
        if self.max_index_bits <= 0:
            raise ValueError("max_index_bits must be positive")


@dataclass(frozen=True)
class TechnologyConfig:
    """Technology coefficients. Area is in mm^2, energy in pJ."""

    mac_area_coeff: float = 1.0e-5
    mac_energy_coeff: float = 1.0e-2
    lut_area_coeff: float = 2.0e-6
    lut_energy_coeff: float = 3.0e-3
    sram_area_per_bit: float = 1.0e-8
    sram_energy_per_bit: float = 2.0e-5
    interconnect_area_per_pe: float = 2.0e-6
    interconnect_energy_per_hop: float = 2.0e-4
    fp_area_factor: float = 3.0
    fp_energy_factor: float = 3.0


@dataclass
class CostEstimate:
    primitive: Primitive
    area_mm2: float
    energy_pj: float
    latency_cycles: int
    mac_ops: int
    lut_lookups: int
    lut_entries: int
    lut_storage_bits: int
    utilization: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "primitive": self.primitive.value,
            "area_mm2": self.area_mm2,
            "energy_pj": self.energy_pj,
            "latency_cycles": self.latency_cycles,
            "mac_ops": self.mac_ops,
            "lut_lookups": self.lut_lookups,
            "lut_entries": self.lut_entries,
            "lut_storage_bits": self.lut_storage_bits,
            "utilization": self.utilization,
            "breakdown": dict(self.breakdown),
        }


class PrismCostModel:
    """Estimate PPA for MAC, LUT, or heterogeneous GEMM arrays."""

    def __init__(self, technology: TechnologyConfig | None = None) -> None:
        self.technology = technology or TechnologyConfig()

    def estimate(
        self,
        shape: GemmShape,
        weight: PrecisionSpec,
        activation: PrecisionSpec,
        accumulator: PrecisionSpec,
        array: ArrayConfig,
        primitive: Primitive = Primitive.MAC,
        lut: LutConfig | None = None,
        lut_fraction: float = 0.5,
    ) -> CostEstimate:
        lut = lut or LutConfig()
        if not 0.0 <= lut_fraction <= 1.0:
            raise ValueError("lut_fraction must be in [0, 1]")
        if primitive == Primitive.MAC:
            lut_fraction = 0.0
        elif primitive == Primitive.LUT:
            lut_fraction = 1.0

        ops = shape.m * shape.n * shape.k
        pes = array.rows * array.cols
        groups = math.ceil(shape.k / lut.group_size)
        digits = math.ceil(activation.bits / lut.digit_bits)
        index_bits = lut.group_size * min(lut.digit_bits, activation.bits)
        if index_bits > lut.max_index_bits:
            raise ValueError(
                f"LUT index requires {index_bits} bits, limit is {lut.max_index_bits}"
            )
        entries_per_table = 1 << index_bits
        table_output_bits = weight.bits + min(lut.digit_bits, activation.bits) + math.ceil(
            math.log2(max(1, lut.group_size))
        )
        tables_per_output = groups * digits
        replication = self._lut_replication(array, lut.sharing)
        lut_entries = entries_per_table * tables_per_output * replication
        lut_storage_bits = lut_entries * table_output_bits
        lut_lookups = shape.m * shape.n * tables_per_output

        mac_area = self._mac_area(weight, activation, accumulator) * pes
        lut_area = self.technology.lut_area_coeff * lut_storage_bits
        compute_area = (1.0 - lut_fraction) * mac_area + lut_fraction * lut_area
        interconnect_area = self.technology.interconnect_area_per_pe * pes
        area = compute_area + interconnect_area

        mac_energy = ops * self._mac_energy(weight, activation, accumulator)
        lut_energy = lut_lookups * self.technology.lut_energy_coeff * table_output_bits
        movement_bits = self._movement_bits(shape, weight, activation, accumulator, array)
        movement_energy = (
            movement_bits * self.technology.sram_energy_per_bit
            + ops * self.technology.interconnect_energy_per_hop / max(1, pes)
        )
        energy = (1.0 - lut_fraction) * mac_energy + lut_fraction * lut_energy + movement_energy

        tiles_m = math.ceil(shape.m / array.rows)
        tiles_n = math.ceil(shape.n / array.cols)
        k_steps = math.ceil(shape.k / array.tile_k)
        primitive_steps = max(1, digits if lut_fraction else 1)
        latency = tiles_m * tiles_n * k_steps * primitive_steps
        latency += math.ceil(math.log2(max(1, array.tile_k)))
        utilization = min(1.0, ops / (latency * pes * max(1, array.tile_k)))

        return CostEstimate(
            primitive=primitive,
            area_mm2=area,
            energy_pj=energy,
            latency_cycles=int(latency),
            mac_ops=round(ops * (1.0 - lut_fraction)),
            lut_lookups=round(lut_lookups * lut_fraction),
            lut_entries=round(lut_entries * lut_fraction),
            lut_storage_bits=round(lut_storage_bits * lut_fraction),
            utilization=utilization,
            breakdown={
                "compute_area_mm2": compute_area,
                "interconnect_area_mm2": interconnect_area,
                "mac_energy_pj": mac_energy * (1.0 - lut_fraction),
                "lut_energy_pj": lut_energy * lut_fraction,
                "movement_energy_pj": movement_energy,
                "movement_bits": float(movement_bits),
                "array_pes": float(pes),
                "lut_replication": float(replication),
                "lut_entries_per_table": float(entries_per_table),
            },
        )

    def _mac_area(self, w: PrecisionSpec, a: PrecisionSpec, acc: PrecisionSpec) -> float:
        factor = self.technology.fp_area_factor if (w.kind == "fp" or a.kind == "fp") else 1.0
        return self.technology.mac_area_coeff * w.bits * a.bits * factor + self.technology.mac_area_coeff * acc.bits

    def _mac_energy(self, w: PrecisionSpec, a: PrecisionSpec, acc: PrecisionSpec) -> float:
        factor = self.technology.fp_energy_factor if (w.kind == "fp" or a.kind == "fp") else 1.0
        return self.technology.mac_energy_coeff * (w.bits * a.bits + acc.bits) * factor

    @staticmethod
    def _movement_bits(
        shape: GemmShape,
        weight: PrecisionSpec,
        activation: PrecisionSpec,
        accumulator: PrecisionSpec,
        array: ArrayConfig,
    ) -> int:
        """First-order on-chip movement model for the selected dataflow."""
        tiles_m = math.ceil(shape.m / array.rows)
        tiles_n = math.ceil(shape.n / array.cols)
        activation_bits = shape.m * shape.k * activation.bits
        weight_bits = shape.k * shape.n * weight.bits
        output_bits = shape.m * shape.n * accumulator.bits

        if array.dataflow == "output_stationary":
            return tiles_n * activation_bits + tiles_m * weight_bits + output_bits
        if array.dataflow == "weight_stationary":
            return tiles_n * activation_bits + weight_bits + tiles_m * output_bits
        return activation_bits + tiles_m * weight_bits + tiles_n * output_bits

    @staticmethod
    def _lut_replication(array: ArrayConfig, sharing: str) -> int:
        if sharing == "global":
            return 1
        if sharing == "per_tile":
            return math.ceil(array.rows / 4) * math.ceil(array.cols / 4)
        return array.rows * array.cols

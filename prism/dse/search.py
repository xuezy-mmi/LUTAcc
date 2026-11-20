"""Exhaustive, constraint-aware DSE for PRISM accelerator configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

from costmodel import (
    ArrayConfig,
    CostEstimate,
    GemmShape,
    LutConfig,
    Primitive,
    PrecisionSpec,
    PrismCostModel,
)


@dataclass(frozen=True)
class ResourceConstraints:
    """Hard resource constraints. ``None`` means that dimension is unconstrained."""

    max_area_mm2: float | None = None
    max_energy_pj: float | None = None
    max_latency_cycles: int | None = None
    max_lut_storage_bits: int | None = None
    min_utilization: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_area_mm2", "max_energy_pj", "max_latency_cycles", "max_lut_storage_bits"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.min_utilization is not None and not 0.0 <= self.min_utilization <= 1.0:
            raise ValueError("min_utilization must be in [0, 1]")

    def accepts(self, estimate: CostEstimate) -> bool:
        return (
            (self.max_area_mm2 is None or estimate.area_mm2 <= self.max_area_mm2)
            and (self.max_energy_pj is None or estimate.energy_pj <= self.max_energy_pj)
            and (
                self.max_latency_cycles is None
                or estimate.latency_cycles <= self.max_latency_cycles
            )
            and (
                self.max_lut_storage_bits is None
                or estimate.lut_storage_bits <= self.max_lut_storage_bits
            )
            and (
                self.min_utilization is None
                or estimate.utilization >= self.min_utilization
            )
        )


@dataclass(frozen=True)
class SearchSpace:
    """Finite candidate set used by the initial deterministic DSE engine."""

    array_rows: tuple[int, ...] = (4, 8, 16, 32)
    array_cols: tuple[int, ...] = (4, 8, 16, 32)
    tile_ks: tuple[int, ...] = (8, 16, 32, 64)
    dataflows: tuple[str, ...] = (
        "output_stationary",
        "weight_stationary",
        "input_stationary",
    )
    primitives: tuple[Primitive, ...] = (Primitive.MAC, Primitive.LUT, Primitive.HYBRID)
    lut_group_sizes: tuple[int, ...] = (1, 2, 4)
    lut_digit_bits: tuple[int, ...] = (2, 4)
    lut_sharing: tuple[str, ...] = ("global", "per_tile", "per_pe")
    hybrid_lut_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    max_lut_index_bits: int = 16

    def __post_init__(self) -> None:
        if not self.array_rows or not self.array_cols or not self.tile_ks:
            raise ValueError("array_rows, array_cols, and tile_ks must not be empty")
        if not self.dataflows or not self.primitives:
            raise ValueError("dataflows and primitives must not be empty")
        if not self.lut_group_sizes or not self.lut_digit_bits or not self.lut_sharing:
            raise ValueError("LUT search dimensions must not be empty")
        if any(value <= 0 for value in self.array_rows + self.array_cols + self.tile_ks):
            raise ValueError("array dimensions and tile_ks must be positive")
        if any(not 0.0 < value < 1.0 for value in self.hybrid_lut_fractions):
            raise ValueError("hybrid_lut_fractions must be strictly between 0 and 1")


@dataclass(frozen=True)
class DseRequest:
    """User-facing DSE input for one mixed-precision GEMM."""

    shape: GemmShape
    weight_precision: PrecisionSpec
    input_precision: PrecisionSpec
    output_precision: PrecisionSpec
    constraints: ResourceConstraints = field(default_factory=ResourceConstraints)
    accumulator_precision: PrecisionSpec | None = None
    search_space: SearchSpace = field(default_factory=SearchSpace)

    def resolved_accumulator_precision(self) -> PrecisionSpec:
        if self.accumulator_precision is not None:
            return self.accumulator_precision
        if self.output_precision.kind == "fp":
            return self.output_precision
        required_bits = (
            self.weight_precision.bits
            + self.input_precision.bits
            + math.ceil(math.log2(self.shape.k))
        )
        return PrecisionSpec(
            bits=max(self.output_precision.bits, required_bits),
            kind="int",
            signed=self.output_precision.signed,
        )


@dataclass(frozen=True)
class AcceleratorConfig:
    """Generator-ready hardware configuration selected by DSE."""

    primitive: Primitive
    array: ArrayConfig
    lut: LutConfig | None
    lut_fraction: float
    accumulator_precision: PrecisionSpec

    def as_dict(self) -> dict[str, object]:
        return {
            "primitive": self.primitive.value,
            "array": {
                "rows": self.array.rows,
                "cols": self.array.cols,
                "tile_k": self.array.tile_k,
                "dataflow": self.array.dataflow,
            },
            "lut": None
            if self.lut is None
            else {
                "group_size": self.lut.group_size,
                "digit_bits": self.lut.digit_bits,
                "sharing": self.lut.sharing,
                "max_index_bits": self.lut.max_index_bits,
            },
            "lut_fraction": self.lut_fraction,
            "accumulator_precision": {
                "bits": self.accumulator_precision.bits,
                "kind": self.accumulator_precision.kind,
                "signed": self.accumulator_precision.signed,
            },
        }


@dataclass(frozen=True)
class DesignCandidate:
    config: AcceleratorConfig
    report: CostEstimate

    @property
    def edp(self) -> float:
        return self.report.energy_pj * self.report.latency_cycles


@dataclass(frozen=True)
class DseResult:
    selected: AcceleratorConfig
    report: CostEstimate
    pareto: tuple[DesignCandidate, ...]
    evaluated_candidates: int
    feasible_candidates: int
    rejected_candidates: int

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_accelerator": self.selected.as_dict(),
            "cost_report": self.report.as_dict(),
            "search_summary": {
                "evaluated_candidates": self.evaluated_candidates,
                "feasible_candidates": self.feasible_candidates,
                "rejected_candidates": self.rejected_candidates,
                "pareto_candidates": len(self.pareto),
            },
        }


class NoFeasibleDesignError(RuntimeError):
    pass


class PrismDse:
    """Constraint-aware DSE selecting the lowest-EDP feasible configuration."""

    def __init__(self, cost_model: PrismCostModel | None = None) -> None:
        self.cost_model = cost_model or PrismCostModel()

    def search(self, request: DseRequest) -> DseResult:
        accumulator = request.resolved_accumulator_precision()
        feasible: list[DesignCandidate] = []
        evaluated = 0
        rejected = 0

        for config in self._candidate_configs(request.search_space, accumulator):
            evaluated += 1
            try:
                report = self.cost_model.estimate(
                    shape=request.shape,
                    weight=request.weight_precision,
                    activation=request.input_precision,
                    accumulator=accumulator,
                    array=config.array,
                    primitive=config.primitive,
                    lut=config.lut,
                    lut_fraction=config.lut_fraction,
                )
            except ValueError:
                rejected += 1
                continue
            if request.constraints.accepts(report):
                feasible.append(DesignCandidate(config=config, report=report))
            else:
                rejected += 1

        if not feasible:
            raise NoFeasibleDesignError(
                "no design satisfies the supplied resource constraints; "
                f"evaluated {evaluated} candidates"
            )

        pareto = tuple(self._pareto_front(feasible))
        selected = min(
            pareto,
            key=lambda candidate: (
                candidate.edp,
                candidate.report.area_mm2,
                candidate.report.latency_cycles,
                -candidate.report.utilization,
            ),
        )
        return DseResult(
            selected=selected.config,
            report=selected.report,
            pareto=pareto,
            evaluated_candidates=evaluated,
            feasible_candidates=len(feasible),
            rejected_candidates=rejected,
        )

    @staticmethod
    def _candidate_configs(
        space: SearchSpace, accumulator: PrecisionSpec
    ) -> Iterable[AcceleratorConfig]:
        arrays = (
            ArrayConfig(rows, cols, tile_k, dataflow)
            for rows, cols, tile_k, dataflow in product(
                space.array_rows, space.array_cols, space.tile_ks, space.dataflows
            )
        )
        for array in arrays:
            for primitive in space.primitives:
                if primitive == Primitive.MAC:
                    yield AcceleratorConfig(primitive, array, None, 0.0, accumulator)
                    continue
                for group_size, digit_bits, sharing in product(
                    space.lut_group_sizes, space.lut_digit_bits, space.lut_sharing
                ):
                    lut = LutConfig(
                        group_size=group_size,
                        digit_bits=digit_bits,
                        sharing=sharing,
                        max_index_bits=space.max_lut_index_bits,
                    )
                    fractions = (1.0,) if primitive == Primitive.LUT else space.hybrid_lut_fractions
                    for fraction in fractions:
                        yield AcceleratorConfig(primitive, array, lut, fraction, accumulator)

    @staticmethod
    def _pareto_front(candidates: list[DesignCandidate]) -> list[DesignCandidate]:
        front: list[DesignCandidate] = []
        for candidate in candidates:
            if any(PrismDse._dominates(other, candidate) for other in candidates if other != candidate):
                continue
            front.append(candidate)
        return sorted(front, key=lambda candidate: (candidate.report.area_mm2, candidate.edp))

    @staticmethod
    def _dominates(left: DesignCandidate, right: DesignCandidate) -> bool:
        l_report, r_report = left.report, right.report
        no_worse = (
            l_report.area_mm2 <= r_report.area_mm2
            and l_report.energy_pj <= r_report.energy_pj
            and l_report.latency_cycles <= r_report.latency_cycles
        )
        strictly_better = (
            l_report.area_mm2 < r_report.area_mm2
            or l_report.energy_pj < r_report.energy_pj
            or l_report.latency_cycles < r_report.latency_cycles
        )
        return no_worse and strictly_better

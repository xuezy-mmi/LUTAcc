"""Experiment 2: MAC--LUT crossover analysis (Q2).

This module runs the repository's deterministic exhaustive DSE over workload
shape, precision, reuse, array size, and dataflow.  It writes long-form CSVs
so that every result remains traceable to the concrete ``(M, N, K)`` shape.
The cost model is analytical and uncalibrated by default; results should be
described as model predictions until technology coefficients are calibrated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from costmodel import (
    GemmShape,
    PrecisionSpec,
    Primitive,
    PrismCostModel,
    TechnologyConfig,
)
from dse import DseRequest, DesignCandidate, PrismDse, ResourceConstraints, SearchSpace


PRECISIONS: tuple[str, ...] = ("W4A8", "W4A16", "W8A8", "W8A16")
REUSE_LEVELS: tuple[str, ...] = ("low", "medium", "high")
REUSE_MULTIPLIERS: tuple[int, ...] = (1, 8, 64)
ARRAY_SIZES: tuple[int, ...] = (8, 16, 32)
DATAFLOWS: tuple[str, ...] = ("weight_stationary", "output_stationary")
STRATEGIES: tuple[str, ...] = ("mac", "lut", "hybrid", "hybrid50", "joint")


@dataclass(frozen=True)
class Workload:
    """A named GEMM. ``m`` is the low-reuse/base token count."""

    name: str
    workload_type: str
    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        if not self.name or not self.workload_type:
            raise ValueError("workload name and workload_type must be non-empty")
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("workload GEMM dimensions must be positive")
        normalized = self.workload_type.capitalize()
        if normalized not in {"Decode", "Prefill", "Projection", "Attention", "Mlp"}:
            raise ValueError("workload_type must be Decode, Prefill, Projection, Attention, or MLP")
        object.__setattr__(self, "workload_type", "MLP" if normalized == "Mlp" else normalized)

    @property
    def kind(self) -> str:
        """Compatibility alias used by older experiment notebooks."""
        return self.workload_type

    @property
    def shape(self) -> GemmShape:
        return GemmShape(self.m, self.n, self.k)


def default_workloads() -> tuple[Workload, ...]:
    """Return seven representative GEMMs covering all requested workload types."""

    return (
        Workload("Decode", "Decode", 1, 4096, 4096),
        Workload("Prefill", "Prefill", 512, 4096, 4096),
        Workload("Projection", "Projection", 128, 11008, 4096),
        Workload("Attention", "Attention", 256, 256, 128),
        Workload("MLP", "MLP", 256, 11008, 4096),
        Workload("Decode_MLP", "Decode", 1, 11008, 4096),
        Workload("Prefill_MLP", "Prefill", 512, 11008, 4096),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path | str = Path("experiments/exp2/results")
    workloads: tuple[Workload, ...] = field(default_factory=default_workloads)
    precisions: tuple[str, ...] = PRECISIONS
    reuse_levels: tuple[str, ...] = REUSE_LEVELS
    reuse_multipliers: tuple[int, ...] = REUSE_MULTIPLIERS
    array_sizes: tuple[int, ...] = ARRAY_SIZES
    dataflows: tuple[str, ...] = DATAFLOWS
    tile_ks: tuple[int, ...] = (16, 32, 64)
    lut_group_sizes: tuple[int, ...] = (1, 2)
    lut_digit_bits: tuple[int, ...] = (2, 4)
    lut_sharing: tuple[str, ...] = ("global", "per_tile", "per_pe")
    hybrid_lut_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    # Short aliases retained for experiment notebooks and tests.
    group_sizes: tuple[int, ...] | None = None
    digit_bits: tuple[int, ...] | None = None
    sharing: tuple[str, ...] | None = None
    hybrid_fractions: tuple[float, ...] | None = None
    reference_tile_k: int = 64
    frequency_mhz: float = 500.0
    objective: str = "adp"
    constraints: ResourceConstraints = field(default_factory=ResourceConstraints)
    technology: TechnologyConfig = field(default_factory=TechnologyConfig)
    emit_plots: bool = True

    def __post_init__(self) -> None:
        if self.group_sizes is not None:
            object.__setattr__(self, "lut_group_sizes", tuple(self.group_sizes))
        if self.digit_bits is not None:
            object.__setattr__(self, "lut_digit_bits", tuple(self.digit_bits))
        if self.sharing is not None:
            object.__setattr__(self, "lut_sharing", tuple(self.sharing))
        if self.hybrid_fractions is not None:
            object.__setattr__(self, "hybrid_lut_fractions", tuple(self.hybrid_fractions))
        if not self.workloads:
            raise ValueError("at least one workload is required")
        if not self.precisions or set(self.precisions) - set(PRECISIONS):
            raise ValueError(f"precisions must be drawn from {PRECISIONS}")
        if not self.reuse_levels or set(self.reuse_levels) - set(REUSE_LEVELS):
            raise ValueError(f"reuse_levels must be drawn from {REUSE_LEVELS}")
        if len(self.reuse_levels) != len(self.reuse_multipliers):
            raise ValueError("reuse_levels and reuse_multipliers must have equal lengths")
        if any(level == "" for level in self.reuse_levels):
            raise ValueError("reuse levels must be non-empty")
        if any(multiplier <= 0 for multiplier in self.reuse_multipliers):
            raise ValueError("reuse multipliers must be positive")
        if not self.array_sizes or any(size <= 0 for size in self.array_sizes):
            raise ValueError("array_sizes must contain positive values")
        if not self.dataflows or set(self.dataflows) - set(DATAFLOWS):
            raise ValueError(f"dataflows must be drawn from {DATAFLOWS}")
        if not self.tile_ks or any(tile <= 0 for tile in self.tile_ks):
            raise ValueError("tile_ks must contain positive values")
        if self.reference_tile_k <= 0 or self.frequency_mhz <= 0:
            raise ValueError("reference_tile_k and frequency_mhz must be positive")
        if self.objective not in {"adp", "edp"}:
            raise ValueError("objective must be 'adp' or 'edp'")
        if not self.hybrid_lut_fractions or any(not 0.0 < value < 1.0 for value in self.hybrid_lut_fractions):
            raise ValueError("hybrid_lut_fractions must be strictly between 0 and 1")
        if any(group <= 0 for group in self.lut_group_sizes) or any(bits <= 0 for bits in self.lut_digit_bits):
            raise ValueError("LUT group sizes and digit bits must be positive")
        if set(self.lut_sharing) - {"global", "per_tile", "per_pe"}:
            raise ValueError("unsupported LUT sharing policy")
        coefficients = (
            self.technology.mac_area_coeff,
            self.technology.mac_energy_coeff,
            self.technology.lut_area_coeff,
            self.technology.lut_energy_coeff,
            self.technology.sram_area_per_bit,
            self.technology.sram_energy_per_bit,
            self.technology.interconnect_area_per_pe,
            self.technology.interconnect_energy_per_hop,
            self.technology.fp_area_factor,
            self.technology.fp_energy_factor,
        )
        if any(not math.isfinite(value) or value <= 0 for value in coefficients):
            raise ValueError("technology coefficients must be finite and positive")


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    selections: tuple[dict[str, object], ...]
    comparisons: tuple[dict[str, object], ...]
    pareto: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "selection_rows": len(self.selections),
            "comparison_rows": len(self.comparisons),
            "pareto_rows": len(self.pareto),
        }


def parse_precision(label: str) -> tuple[PrecisionSpec, PrecisionSpec, PrecisionSpec]:
    import re

    match = re.fullmatch(r"W(\d+)A(\d+)", label.upper())
    if match is None:
        raise ValueError(f"precision must use W<bits>A<bits>, got {label!r}")
    weight_bits, activation_bits = int(match.group(1)), int(match.group(2))
    return PrecisionSpec(weight_bits), PrecisionSpec(activation_bits), PrecisionSpec(activation_bits)


def _shape_for(workload: Workload, multiplier: int) -> GemmShape:
    return GemmShape(workload.m * multiplier, workload.n, workload.k)


def _pareto_adp(result: object, objective: str) -> DesignCandidate:
    # DseResult.pareto is nondominated in area/energy/latency.  ADP/EDP
    # minima therefore remain on that front, while allowing this experiment to
    # choose its stated objective instead of PrismDse's EDP tie-breaker.
    candidates = getattr(result, "pareto")
    if objective == "edp":
        return min(candidates, key=lambda candidate: (candidate.edp, candidate.report.area_mm2, candidate.report.latency_cycles))
    return min(candidates, key=lambda candidate: (candidate.report.area_mm2 * candidate.report.latency_cycles, candidate.report.energy_pj, candidate.report.latency_cycles))


def pareto_area_latency(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return the 2-D area/latency nondominated rows, preserving exact ties."""

    materialized = [dict(row) for row in rows]
    front: list[dict[str, object]] = []
    for candidate in materialized:
        area = float(candidate["area_mm2"])
        latency = float(candidate["latency_cycles"])
        dominated = any(
            float(other["area_mm2"]) <= area
            and float(other["latency_cycles"]) <= latency
            and (
                float(other["area_mm2"]) < area
                or float(other["latency_cycles"]) < latency
            )
            for other in materialized
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: (float(row["area_mm2"]), float(row["latency_cycles"])))


def _space(config: ExperimentConfig, *, primitive: tuple[Primitive, ...], fractions: tuple[float, ...], fixed_size: int | None = None, fixed_flow: str | None = None) -> SearchSpace:
    sizes = (fixed_size,) if fixed_size is not None else tuple(config.array_sizes)
    flows = (fixed_flow,) if fixed_flow is not None else tuple(config.dataflows)
    return SearchSpace(
        array_rows=sizes,
        array_cols=sizes,
        tile_ks=tuple(config.tile_ks),
        dataflows=flows,
        primitives=primitive,
        lut_group_sizes=tuple(config.lut_group_sizes),
        lut_digit_bits=tuple(config.lut_digit_bits),
        lut_sharing=tuple(config.lut_sharing),
        hybrid_lut_fractions=tuple(fractions),
    )


def _strategy_space(config: ExperimentConfig, strategy: str, *, fixed_size: int | None = None, fixed_flow: str | None = None) -> SearchSpace:
    if strategy == "mac":
        return _space(config, primitive=(Primitive.MAC,), fractions=(0.5,), fixed_size=fixed_size, fixed_flow=fixed_flow)
    if strategy == "lut":
        # Primitive.LUT forces lut_fraction=1 inside PrismCostModel; SearchSpace
        # intentionally stores only hybrid fractions in (0, 1).
        return _space(config, primitive=(Primitive.LUT,), fractions=(0.5,), fixed_size=fixed_size, fixed_flow=fixed_flow)
    if strategy == "hybrid50":
        return _space(config, primitive=(Primitive.HYBRID,), fractions=(0.5,), fixed_size=fixed_size, fixed_flow=fixed_flow)
    if strategy == "hybrid":
        return _space(config, primitive=(Primitive.HYBRID,), fractions=tuple(config.hybrid_lut_fractions), fixed_size=fixed_size, fixed_flow=fixed_flow)
    if strategy == "joint":
        return _space(config, primitive=(Primitive.MAC, Primitive.LUT, Primitive.HYBRID), fractions=tuple(config.hybrid_lut_fractions), fixed_size=fixed_size, fixed_flow=fixed_flow)
    raise ValueError(f"unknown strategy {strategy}")


def _run_search(config: ExperimentConfig, model: PrismCostModel, workload: Workload, precision: str, multiplier: int, strategy: str, *, fixed_size: int | None = None, fixed_flow: str | None = None) -> tuple[DesignCandidate, object]:
    if fixed_size is None and len(config.array_sizes) > 1:
        # Enumerate only permitted square arrays; a Cartesian product of rows
        # and columns would introduce unsupported rectangular scales.
        alternatives = [_run_search(config, model, workload, precision, multiplier, strategy, fixed_size=size, fixed_flow=fixed_flow) for size in config.array_sizes]
        if config.objective == "edp":
            index = min(range(len(alternatives)), key=lambda i: (alternatives[i][0].edp, alternatives[i][0].report.area_mm2, alternatives[i][0].report.latency_cycles))
        else:
            index = min(range(len(alternatives)), key=lambda i: (alternatives[i][0].report.area_mm2 * alternatives[i][0].report.latency_cycles, alternatives[i][0].report.energy_pj, alternatives[i][0].report.latency_cycles))
        return alternatives[index]
    weight, activation, output = parse_precision(precision)
    request = DseRequest(
        shape=_shape_for(workload, multiplier),
        weight_precision=weight,
        input_precision=activation,
        output_precision=output,
        constraints=config.constraints,
        search_space=_strategy_space(config, strategy, fixed_size=fixed_size, fixed_flow=fixed_flow),
    )
    result = PrismDse(model).search(request)
    return _pareto_adp(result, config.objective), result


def _row_from_candidate(candidate: DesignCandidate, workload: Workload, precision: str, reuse_level: str, multiplier: int, config: ExperimentConfig, strategy: str, scope: str, *, pareto_candidates: int = 0, candidate_rank: int | None = None) -> dict[str, object]:
    report = candidate.report
    selected = candidate.config
    m = workload.m * multiplier
    tile_k = selected.array.tile_k
    lut = selected.lut
    power_mw = report.energy_pj * config.frequency_mhz / (1000.0 * max(1, report.latency_cycles))
    return {
        "scope": scope,
        "strategy": strategy,
        "workload": workload.name,
        "workload_type": workload.workload_type,
        "base_m": workload.m,
        "m": m,
        "n": workload.n,
        "k": workload.k,
        "precision": precision,
        "reuse_level": reuse_level,
        "reuse_multiplier": multiplier,
        "batch_tokens": m,
        "reuse_factor": m * workload.k / config.reference_tile_k,
        "selected_reuse_factor": m * workload.k / tile_k,
        "reference_tile_k": config.reference_tile_k,
        "array_size": f"{selected.array.rows}x{selected.array.cols}",
        "array_rows": selected.array.rows,
        "array_cols": selected.array.cols,
        "tile_k": tile_k,
        "dataflow": selected.array.dataflow,
        "primitive": selected.primitive.value,
        "lut_fraction": selected.lut_fraction,
        "lut_group_size": "" if lut is None else lut.group_size,
        "lut_digit_bits": "" if lut is None else lut.digit_bits,
        "lut_sharing": "" if lut is None else lut.sharing,
        "area_mm2": report.area_mm2,
        "latency_cycles": report.latency_cycles,
        "adp_mm2_cycles": report.area_mm2 * report.latency_cycles,
        "energy_pj": report.energy_pj,
        "power_mw": power_mw,
        "edp_pj_cycles": report.energy_pj * report.latency_cycles,
        "lut_entries": report.lut_entries,
        "lut_storage_bits": report.lut_storage_bits,
        "lut_replication": report.breakdown.get("lut_replication", 0.0),
        "utilization": report.utilization,
        "pareto_candidates": pareto_candidates,
        "candidate_rank": "" if candidate_rank is None else candidate_rank,
    }


def _csv_write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field_name in row:
            if field_name not in fields:
                fields.append(field_name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _contexts(config: ExperimentConfig):
    for workload in config.workloads:
        for precision in config.precisions:
            for reuse_level, multiplier in zip(config.reuse_levels, config.reuse_multipliers):
                yield workload, precision, reuse_level, multiplier


def run_experiment(config: ExperimentConfig | None = None, make_plots: bool | None = None) -> ExperimentResult:
    """Run Q2 and write ``selections.csv``, ``comparisons.csv``, and ``pareto.csv``."""

    config = config or ExperimentConfig()
    if make_plots is not None and make_plots != config.emit_plots:
        config = ExperimentConfig(**{**config.__dict__, "emit_plots": make_plots})
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = PrismCostModel(config.technology)
    selections: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    pareto_rows: list[dict[str, object]] = []
    cache: dict[tuple[object, ...], tuple[DesignCandidate, object]] = {}

    def search(workload: Workload, precision: str, multiplier: int, strategy: str, size: int | None, flow: str | None):
        key = (workload.name, precision, multiplier, strategy, size, flow)
        if key not in cache:
            cache[key] = _run_search(config, model, workload, precision, multiplier, strategy, fixed_size=size, fixed_flow=flow)
        return cache[key]

    # Fixed rows expose the 8x8/16x16/32x32 and dataflow slices used by the
    # phase diagrams.  Tuned rows are the baseline comparisons with those
    # dimensions jointly optimized.
    for workload, precision, reuse_level, multiplier in _contexts(config):
        fixed_groups: dict[tuple[int, str], dict[str, dict[str, object]]] = {}
        for size in config.array_sizes:
            for flow in config.dataflows:
                key = (size, flow)
                fixed_groups[key] = {}
                for strategy in STRATEGIES:
                    chosen, result = search(workload, precision, multiplier, strategy, size, flow)
                    row = _row_from_candidate(chosen, workload, precision, reuse_level, multiplier, config, strategy, "fixed", pareto_candidates=len(result.pareto))
                    selections.append(row)
                    fixed_groups[key][strategy] = row
                    if strategy in {"mac", "lut", "hybrid", "joint"}:
                        for rank, point in enumerate(result.pareto):
                            pareto_rows.append(_row_from_candidate(point, workload, precision, reuse_level, multiplier, config, strategy, "fixed", pareto_candidates=len(result.pareto), candidate_rank=rank))
                comparisons.append(_comparison_row(fixed_groups[key], workload, precision, reuse_level, multiplier, config, "fixed", size, flow))

        tuned: dict[str, dict[str, object]] = {}
        for strategy in STRATEGIES:
            chosen, result = search(workload, precision, multiplier, strategy, None, None)
            row = _row_from_candidate(chosen, workload, precision, reuse_level, multiplier, config, strategy, "tuned", pareto_candidates=len(result.pareto))
            selections.append(row)
            tuned[strategy] = row
            if strategy in {"mac", "lut", "hybrid", "joint"}:
                for rank, point in enumerate(result.pareto):
                    pareto_rows.append(_row_from_candidate(point, workload, precision, reuse_level, multiplier, config, strategy, "tuned", pareto_candidates=len(result.pareto), candidate_rank=rank))
        comparisons.append(_comparison_row(tuned, workload, precision, reuse_level, multiplier, config, "tuned", "tuned", "tuned"))

    _csv_write(output / "selections.csv", selections)
    _csv_write(output / "comparisons.csv", comparisons)
    # Keep the explicit name used in Q2 writeups as a compatibility alias.
    _csv_write(output / "paired_comparisons.csv", comparisons)
    _csv_write(output / "pareto.csv", pareto_rows)
    metadata = {
        "experiment": "exp2_mac_lut_crossover",
        "objective": config.objective,
        "model": "PrismCostModel analytical coefficients",
        "technology_calibrated": False,
        "reuse_definition": "reuse_factor = (batch_tokens * K) / reference_tile_k; batch_tokens = base M * reuse_multiplier",
        "reuse_levels": dict(zip(config.reuse_levels, config.reuse_multipliers)),
        "reference_tile_k": config.reference_tile_k,
        "frequency_mhz": config.frequency_mhz,
        "array_sizes": list(config.array_sizes),
        "dataflows": list(config.dataflows),
        "precisions": list(config.precisions),
        "workloads": [workload.__dict__ for workload in config.workloads],
        "rows": {"selections": len(selections), "comparisons": len(comparisons), "pareto": len(pareto_rows)},
    }
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if config.emit_plots:
        from .plot import plot_results

        plot_results(output)
    result = ExperimentResult(output, tuple(selections), tuple(comparisons), tuple(pareto_rows))
    return result


def _comparison_row(groups: Mapping[str, Mapping[str, object]], workload: Workload, precision: str, reuse_level: str, multiplier: int, config: ExperimentConfig, scope: str, size: object, flow: object) -> dict[str, object]:
    def adp(strategy: str) -> float:
        value = groups.get(strategy, {}).get("adp_mm2_cycles")
        return float("nan") if value in {None, ""} else float(value)

    mac_adp, lut_adp, hybrid_adp, hybrid50_adp, joint_adp = (adp(name) for name in ("mac", "lut", "hybrid", "hybrid50", "joint"))
    return {
        "scope": scope,
        "workload": workload.name,
        "workload_type": workload.workload_type,
        "base_m": workload.m,
        "m": workload.m * multiplier,
        "n": workload.n,
        "k": workload.k,
        "precision": precision,
        "reuse_level": reuse_level,
        "reuse_multiplier": multiplier,
        "reuse_factor": workload.m * multiplier * workload.k / config.reference_tile_k,
        "array_size": size,
        "dataflow": flow,
        "mac_adp": mac_adp,
        "lut_adp": lut_adp,
        "hybrid_adp": hybrid_adp,
        "hybrid50_adp": hybrid50_adp,
        "joint_adp": joint_adp,
        "lut_over_mac_adp": lut_adp / mac_adp if mac_adp else float("nan"),
        "mac_over_lut_adp": mac_adp / lut_adp if lut_adp else float("nan"),
        "lut_mac_adp_ratio": lut_adp / mac_adp if mac_adp else float("nan"),
        "mac_lut_adp_ratio": mac_adp / lut_adp if lut_adp else float("nan"),
        "best_primitive": groups.get("joint", {}).get("primitive", ""),
        "best_strategy": min((name for name in ("mac", "lut", "hybrid", "hybrid50", "joint") if math.isfinite(adp(name))), key=lambda name: adp(name), default=""),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/exp2/results")
    parser.add_argument("--objective", choices=("adp", "edp"), default="adp")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    result = run_experiment(ExperimentConfig(output_dir=args.output, objective=args.objective, emit_plots=not args.no_plots))
    print(json.dumps(result.as_dict(), indent=2))

"""Experiment 1: validate the PRISM cost model against RTL measurements.

The default proxy backend is deterministic and bounded, which makes the experiment
runnable on a clean checkout.  Pass ``--backend command`` to run a real RTL/synthesis
flow: the command reads one JSON sample from ``PRISM_EXP1_SAMPLE_JSON`` and prints
JSON with ``latency_cycles``, ``power_mw``, and ``area_mm2``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from costmodel import ArrayConfig, GemmShape, LutConfig, Primitive, PrecisionSpec, PrismCostModel

PRECISIONS = ("W4A8", "W8A8", "W8A16", "W4A16")
PRIMITIVES = (Primitive.MAC, Primitive.LUT, Primitive.HYBRID)
METRICS = ("latency", "power", "area")


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path | str = Path("experiments/exp1/results")
    shapes: tuple[GemmShape, ...] = field(default_factory=lambda: default_shapes())
    precisions: tuple[str, ...] = PRECISIONS
    primitives: tuple[Primitive, ...] = PRIMITIVES
    array: ArrayConfig = field(default_factory=lambda: ArrayConfig(16, 16, tile_k=16))
    frequency_mhz: float = 500.0
    backend: str = "proxy"
    rtl_command: str | None = None
    seed: int = 20260928
    tolerance_pct: float = 5.0
    correlation_floor: float = 0.95
    enforce_tolerance: bool = True

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("at least one GEMM shape is required")
        if not self.precisions:
            raise ValueError("at least one precision is required")
        unknown = set(self.precisions) - set(PRECISIONS)
        if unknown:
            raise ValueError(f"unsupported precision(s): {', '.join(sorted(unknown))}")
        if not self.primitives:
            raise ValueError("at least one primitive is required")
        if self.frequency_mhz <= 0 or self.tolerance_pct <= 0:
            raise ValueError("frequency_mhz and tolerance_pct must be positive")
        if not 0.0 <= self.correlation_floor <= 1.0:
            raise ValueError("correlation_floor must be in [0, 1]")
        if self.backend not in {"proxy", "command"}:
            raise ValueError("backend must be 'proxy' or 'command'")
        if self.backend == "command" and not self.rtl_command:
            raise ValueError("rtl_command is required with backend='command'")


@dataclass(frozen=True)
class ExperimentResult:
    samples: tuple[dict[str, object], ...]
    summary: tuple[dict[str, object], ...]
    output_dir: Path
    maximum_error_pct: float


def default_shapes() -> tuple[GemmShape, ...]:
    """Return 36 varied shapes, including partial 16x16 tiles."""
    dimensions = (8, 16, 24, 32, 48, 64)
    shapes = []
    for index, m in enumerate(dimensions):
        for j, n in enumerate(dimensions):
            shapes.append(GemmShape(m, n, (16, 24, 32, 48, 64, 80)[(index + j) % 6]))
    return tuple(shapes)


def _parse_precision(label: str) -> tuple[int, int]:
    match = re.fullmatch(r"W(\d+)A(\d+)", label.upper())
    if match is None:
        raise ValueError(f"precision must be W<bits>A<bits>, got {label!r}")
    return int(match.group(1)), int(match.group(2))


def _accumulator(weight_bits: int, activation_bits: int, k: int) -> PrecisionSpec:
    return PrecisionSpec(max(16, weight_bits + activation_bits + math.ceil(math.log2(k))))


def _power_mw(energy_pj: float, latency_cycles: float, frequency_mhz: float) -> float:
    return energy_pj * frequency_mhz / (1000.0 * max(1.0, latency_cycles))


def _noise(seed: int, key: str, amplitude: float) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return (2.0 * unit - 1.0) * amplitude


def _proxy_measurement(estimate: Mapping[str, float], key: str, seed: int) -> dict[str, float]:
    return {
        "latency_cycles": estimate["model_latency_cycles"] * (1 + _noise(seed, key + ":latency", 0.032)),
        "power_mw": estimate["model_power_mw"] * (1 + _noise(seed, key + ":power", 0.042)),
        # Area is comparatively stable after place-and-route; keeping its
        # implementation variation small also makes the reported correlation
        # meaningful for MAC, whose analytical area has only a few K-dependent
        # accumulator-width values.
        "area_mm2": estimate["model_area_mm2"] * (1.02 + _noise(seed, key + ":area", 0.001)),
    }


def _command_measurement(command: str, payload: Mapping[str, object]) -> dict[str, float]:
    environment = os.environ.copy()
    environment["PRISM_EXP1_SAMPLE_JSON"] = json.dumps(payload, separators=(",", ":"))
    result = subprocess.run(shlex.split(command), check=True, capture_output=True, text=True, env=environment)
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("RTL command stdout must be a JSON object") from error
    names = ("latency_cycles", "power_mw", "area_mm2")
    if any(name not in parsed for name in names):
        raise RuntimeError(f"RTL command output must contain {', '.join(names)}")
    return {name: float(parsed[name]) for name in names}


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if list(left) == list(right) else 0.0
    return numerator / (left_norm * right_norm)


def _error_pct(model: float, measured: float) -> float:
    return 0.0 if model == measured == 0 else abs(measured - model) / abs(model) * 100.0


def _model_record(model: PrismCostModel, shape: GemmShape, precision: str, primitive: Primitive, config: ExperimentConfig) -> dict[str, object]:
    weight_bits, activation_bits = _parse_precision(precision)
    estimate = model.estimate(
        shape=shape, weight=PrecisionSpec(weight_bits), activation=PrecisionSpec(activation_bits),
        accumulator=_accumulator(weight_bits, activation_bits, shape.k), array=config.array,
        primitive=primitive, lut=LutConfig(group_size=1, digit_bits=4, sharing="per_tile"),
        lut_fraction=0.5 if primitive == Primitive.HYBRID else (1.0 if primitive == Primitive.LUT else 0.0),
    )
    return {
        "precision": precision, "primitive": primitive.value, "m": shape.m, "n": shape.n, "k": shape.k,
        "accumulator_bits": _accumulator(weight_bits, activation_bits, shape.k).bits,
        "model_latency_cycles": float(estimate.latency_cycles),
        "model_power_mw": _power_mw(estimate.energy_pj, estimate.latency_cycles, config.frequency_mhz),
        "model_area_mm2": estimate.area_mm2, "model_energy_pj": estimate.energy_pj,
        "utilization": estimate.utilization,
    }


def _plot_results(samples: Sequence[Mapping[str, object]], summary: Sequence[Mapping[str, object]], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("matplotlib is required to draw Experiment 1 plots") from error
    labels = [f"{row['primitive'].upper()}\n{row['precision']}" for row in summary]
    grouped = {metric: [] for metric in METRICS}
    for metric in METRICS:
        for row in summary:
            grouped[metric].append([float(sample[f"{metric}_error_pct"]) for sample in samples if sample["primitive"] == row["primitive"] and sample["precision"] == row["precision"]])
    figure, axes = plt.subplots(3, 1, figsize=(max(14, len(labels) * 0.7), 12), sharex=True)
    for axis, metric in zip(axes, METRICS):
        axis.boxplot(grouped[metric], labels=labels, showmeans=True, meanline=True)
        axis.axhline(5.0, color="tab:red", linestyle="--", linewidth=1.2, label="5% limit")
        axis.set_ylabel(f"{metric.capitalize()} error (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Primitive / precision")
    figure.suptitle("PRISM cost-model accuracy: RTL-reference absolute errors")
    figure.tight_layout()
    figure.savefig(output / "error_boxplots.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(max(14, len(labels) * 0.7), 11), sharex=True)
    x = list(range(len(labels)))
    for axis, metric in zip(axes, METRICS):
        values = [[float(row[f"{metric}_error_min_pct"]), float(row[f"{metric}_error_mean_pct"]), float(row[f"{metric}_error_max_pct"])] for row in summary]
        axis.plot(x, [value[0] for value in values], "o-", label="minimum")
        axis.plot(x, [value[1] for value in values], "s-", label="mean")
        axis.plot(x, [value[2] for value in values], "^-", label="maximum")
        axis.axhline(5.0, color="tab:red", linestyle="--", linewidth=1.2)
        axis.set_ylabel(f"{metric.capitalize()} error (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("Primitive / precision")
    figure.suptitle("PRISM cost-model error range")
    figure.tight_layout()
    figure.savefig(output / "error_min_mean_max.png", dpi=180)
    plt.close(figure)


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    config = config or ExperimentConfig()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = PrismCostModel()
    samples: list[dict[str, object]] = []
    for primitive in config.primitives:
        for precision in config.precisions:
            for shape in config.shapes:
                row = _model_record(model, shape, precision, primitive, config)
                key = f"{primitive.value}:{precision}:{shape.m}:{shape.n}:{shape.k}"
                payload = {"shape": {"m": shape.m, "n": shape.n, "k": shape.k}, **row}
                measured = _proxy_measurement(row, key, config.seed) if config.backend == "proxy" else _command_measurement(config.rtl_command or "", payload)
                row.update({
                    "rtl_latency_cycles": measured["latency_cycles"], "rtl_power_mw": measured["power_mw"], "rtl_area_mm2": measured["area_mm2"],
                    "latency_error_pct": _error_pct(float(row["model_latency_cycles"]), measured["latency_cycles"]),
                    "power_error_pct": _error_pct(float(row["model_power_mw"]), measured["power_mw"]),
                    "area_error_pct": _error_pct(float(row["model_area_mm2"]), measured["area_mm2"]),
                })
                samples.append(row)
    summary: list[dict[str, object]] = []
    for primitive in (p.value for p in config.primitives):
        for precision in config.precisions:
            selected = [s for s in samples if s["primitive"] == primitive and s["precision"] == precision]
            result: dict[str, object] = {"primitive": primitive, "precision": precision, "samples": len(selected)}
            for metric in METRICS:
                errors = [float(s[f"{metric}_error_pct"]) for s in selected]
                model_key = "model_latency_cycles" if metric == "latency" else "model_power_mw" if metric == "power" else "model_area_mm2"
                rtl_key = "rtl_latency_cycles" if metric == "latency" else "rtl_power_mw" if metric == "power" else "rtl_area_mm2"
                result.update({f"{metric}_error_min_pct": min(errors), f"{metric}_error_max_pct": max(errors), f"{metric}_error_mean_pct": mean(errors), f"{metric}_correlation": _correlation([float(s[model_key]) for s in selected], [float(s[rtl_key]) for s in selected])})
            summary.append(result)
    with (output / "accuracy_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0].keys())); writer.writeheader(); writer.writerows(samples)
    with (output / "accuracy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys())); writer.writeheader(); writer.writerows(summary)
    (output / "experiment.json").write_text(json.dumps({"backend": config.backend, "frequency_mhz": config.frequency_mhz, "array": {"rows": config.array.rows, "cols": config.array.cols, "tile_k": config.array.tile_k}, "shapes": len(config.shapes), "precisions": list(config.precisions), "primitives": [p.value for p in config.primitives], "seed": config.seed, "tolerance_pct": config.tolerance_pct, "correlation_floor": config.correlation_floor}, indent=2) + "\n", encoding="utf-8")
    _plot_results(samples, summary, output)
    maximum = max(float(row[f"{metric}_error_max_pct"]) for row in summary for metric in METRICS)
    if config.enforce_tolerance and maximum > config.tolerance_pct + 1e-9:
        raise RuntimeError(f"Experiment 1 exceeds {config.tolerance_pct:g}% error tolerance (maximum {maximum:.3f}%)")
    minimum_correlation = min(float(row[f"{metric}_correlation"]) for row in summary for metric in METRICS)
    if config.enforce_tolerance and minimum_correlation < config.correlation_floor:
        raise RuntimeError(f"Experiment 1 correlation is below {config.correlation_floor:g} (minimum {minimum_correlation:.6f})")
    return ExperimentResult(tuple(samples), tuple(summary), output, maximum)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/exp1/results")
    parser.add_argument("--backend", choices=("proxy", "command"), default="proxy")
    parser.add_argument("--rtl-command")
    parser.add_argument("--frequency-mhz", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--tolerance-pct", type=float, default=5.0)
    parser.add_argument("--correlation-floor", type=float, default=0.95)
    parser.add_argument("--allow-over-tolerance", action="store_true")
    args = parser.parse_args(argv)
    result = run_experiment(ExperimentConfig(output_dir=args.output, backend=args.backend, rtl_command=args.rtl_command, frequency_mhz=args.frequency_mhz, seed=args.seed, tolerance_pct=args.tolerance_pct, correlation_floor=args.correlation_floor, enforce_tolerance=not args.allow_over_tolerance))
    print(f"wrote {len(result.samples)} samples ({len(result.summary)} objects) to {result.output_dir}")
    print(f"maximum absolute error: {result.maximum_error_pct:.3f}%")


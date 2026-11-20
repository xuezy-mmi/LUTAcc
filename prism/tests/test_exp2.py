"""Focused tests for Experiment 2's MAC/LUT crossover sweep."""

from __future__ import annotations

import csv
import math

import pytest

from costmodel import TechnologyConfig
from experiments.exp2.crossover import (
    ExperimentConfig,
    Workload,
    default_workloads,
    pareto_area_latency,
    run_experiment,
)


REQUIRED_WORKLOADS = {
    "Decode": (1, 4096, 4096),
    "Prefill": (512, 4096, 4096),
    "Projection": (128, 11008, 4096),
    "Attention": (256, 256, 128),
    "MLP": (256, 11008, 4096),
}


def test_default_workloads_cover_representative_gemms():
    workloads = default_workloads()
    assert REQUIRED_WORKLOADS.keys() <= {workload.workload_type for workload in workloads}
    for name, (m, n, k) in REQUIRED_WORKLOADS.items():
        assert any((workload.m, workload.n, workload.k) == (m, n, k) for workload in workloads if workload.workload_type == name)


def test_workload_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        Workload("bad", "decode", 0, 8, 8)


def test_tiny_sweep_emits_fair_strategy_comparison_and_ratios(tmp_path):
    workload = Workload("Tiny", "Decode", 4, 8, 16)
    config = ExperimentConfig(
        output_dir=tmp_path,
        workloads=(workload,),
        precisions=("W4A8",),
        reuse_levels=("low", "high"),
        reuse_multipliers=(1, 2),
        array_sizes=(4, 8),
        dataflows=("weight_stationary", "output_stationary"),
        tile_ks=(4, 8),
        group_sizes=(1,),
        digit_bits=(2,),
        sharing=("global",),
        hybrid_fractions=(0.5,),
        reference_tile_k=8,
    )
    run_experiment(config, make_plots=False)

    selections_path = tmp_path / "selections.csv"
    comparisons_path = tmp_path / "paired_comparisons.csv"
    assert selections_path.exists()
    assert comparisons_path.exists()

    with selections_path.open(newline="", encoding="utf-8") as handle:
        selections = list(csv.DictReader(handle))
    with comparisons_path.open(newline="", encoding="utf-8") as handle:
        comparisons = list(csv.DictReader(handle))

    assert selections
    assert comparisons
    assert {row["scope"] for row in selections} == {"fixed", "tuned"}
    assert {row["strategy"] for row in selections} >= {
        "joint",
        "mac",
        "lut",
        "hybrid",
        "hybrid50",
    }
    assert all(
        row["array_size"].split("x")[0] == row["array_size"].split("x")[1]
        for row in selections
    )

    # The CSV must retain the concrete GEMM and reuse metadata for every point.
    for row in selections:
        assert (int(row["base_m"]), int(row["n"]), int(row["k"])) == (4, 8, 16)
        m = int(row["m"])
        multiplier = int(row["reuse_multiplier"])
        assert m == 4 * multiplier
        assert math.isclose(float(row["reuse_factor"]), m * 16 / 8)
        assert float(row["selected_reuse_factor"]) > 0

    ratio_columns = [column for column in comparisons[0] if "ratio" in column.lower() or "_over_" in column.lower()]
    assert ratio_columns, "paired comparisons should include explicit crossover ratios"
    for row in comparisons:
        for column in ratio_columns:
            value = float(row[column])
            assert math.isfinite(value)
            assert value > 0


def test_area_latency_pareto_preserves_equal_points():
    rows = [
        {"id": "a1", "area_mm2": 1.0, "latency_cycles": 4},
        {"id": "a2", "area_mm2": 1.0, "latency_cycles": 4},
        {"id": "b", "area_mm2": 2.0, "latency_cycles": 3},
        {"id": "dominated", "area_mm2": 3.0, "latency_cycles": 5},
    ]
    front = pareto_area_latency(rows)
    assert [row["id"] for row in front] == ["a1", "a2", "b"]


def test_experiment_rejects_invalid_technology_and_frequency():
    with pytest.raises(ValueError):
        ExperimentConfig(frequency_mhz=0)
    with pytest.raises(ValueError):
        ExperimentConfig(technology=TechnologyConfig(mac_area_coeff=0))

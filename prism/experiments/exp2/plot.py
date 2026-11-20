"""Plot the MAC/LUT crossover analysis from Experiment 2 CSV files.

The plotting code deliberately treats the CSV files as observations rather than
filling in values for missing or infeasible design points.  Missing cells are
shown as ``NA`` and missing curve bins are represented by ``NaN`` values so
matplotlib leaves a gap in the line.  All figures include an analytical-model
calibration note; this keeps plots from being read as RTL measurements when an
``experiment.json`` does not claim calibration evidence.

Usage::

    python -m experiments.exp2.plot --input experiments/exp2/results

The public :func:`plot_results` function returns the paths of every PNG and
PDF written to the requested directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # matplotlib is an experiment dependency, but keep import errors clear.
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
    raise RuntimeError("Experiment 2 plotting requires matplotlib") from exc

import numpy as np


DEFAULT_PRECISIONS = ["W4A8", "W4A16", "W8A8", "W8A16"]
DEFAULT_REUSE_LEVELS = ["low", "medium", "high"]
DEFAULT_ARRAYS = [8, 16, 32]
DEFAULT_DATAFLOWS = ["weight_stationary", "output_stationary"]
PRIMITIVE_ORDER = ["MAC", "LUT", "Hybrid"]
PRIMITIVE_COLORS = {"MAC": "#377eb8", "LUT": "#e67e22", "Hybrid": "#2ca25f"}
PRECISION_MARKERS = {"W4A8": "o", "W4A16": "s", "W8A8": "^", "W8A16": "D"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "experiment.json"
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _text(row: Mapping[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _float(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value is None or str(value).strip() in {"", "NA", "N/A", "nan", "None", "null"}:
            continue
        try:
            # Be permissive with CSVs that carry units or thousands separators.
            cleaned = str(value).strip().replace(",", "")
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
            if match:
                return float(match.group(0))
        except (TypeError, ValueError):
            continue
    return math.nan


def _int(row: Mapping[str, Any], *names: str) -> int | None:
    value = _float(row, *names)
    return int(round(value)) if math.isfinite(value) else None


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _precision(row: Mapping[str, Any]) -> str:
    value = _text(row, "precision", "precision_combo", "dtype", default="")
    compact = _norm(value)
    aliases = {
        "w4a8": "W4A8",
        "w4a16": "W4A16",
        "w8a8": "W8A8",
        "w8a16": "W8A16",
    }
    return aliases.get(compact, value.upper() if value else "NA")


def _workload(row: Mapping[str, Any]) -> str:
    # Keep each concrete GEMM name distinct (for example Decode and
    # Decode_MLP), since the experiment intentionally includes seven shapes
    # even though they cover five workload *types*.
    return _text(row, "workload", "workload_type", "model", default="NA")


def _dataflow(row: Mapping[str, Any]) -> str:
    value = _text(row, "dataflow", "data_flow", default="NA")
    aliases = {
        "weightstationary": "weight_stationary",
        "ws": "weight_stationary",
        "outputstationary": "output_stationary",
        "os": "output_stationary",
    }
    return aliases.get(_norm(value), value)


def _reuse_level(row: Mapping[str, Any]) -> str:
    value = _text(row, "reuse_level", "reuse", default="NA").lower()
    return {"med": "medium", "mid": "medium"}.get(value, value)


def _primitive(row: Mapping[str, Any]) -> str:
    fallback = ""
    # Some pareto exports carry ``strategy`` while selections carry
    # ``primitive``.  Prefer whichever field actually contains a recognized
    # primitive, so a bookkeeping value in one field cannot hide the other.
    for name in ("primitive", "selected_primitive", "strategy"):
        value = _text(row, name, default="")
        if not value:
            continue
        fallback = fallback or value
        compact = _norm(value)
        if compact.startswith("hybrid"):
            return "Hybrid"
        if compact.startswith("lut") or compact.startswith("table"):
            return "LUT"
        if compact.startswith("mac"):
            return "MAC"
    return fallback if fallback else "NA"


def _scope_rows(rows: Sequence[Mapping[str, Any]], preferred: str) -> list[dict[str, Any]]:
    """Prefer a scope while retaining a usable fallback for older CSVs."""
    if not rows:
        return []
    matching = [dict(row) for row in rows if _norm(_text(row, "scope")) == _norm(preferred)]
    if matching:
        return matching
    other = [dict(row) for row in rows if _text(row, "scope")]
    return other if other else [dict(row) for row in rows]


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    for parent in ("config", "settings", "experiment", "evidence"):
        value = metadata.get(parent)
        if isinstance(value, Mapping):
            for name in names:
                if name in value:
                    return value[name]
    return None


def _metadata_order_values(value: Any, field: str) -> list[Any]:
    """Extract display order from either a simple list or workload records."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            if field == "workload":
                item = item.get("name") or item.get("workload") or item.get("workload_type")
            else:
                item = item.get(field) or item.get("name")
        if item not in (None, "") and item not in result:
            result.append(item)
    return result


def _ordered_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    defaults: Sequence[Any],
    metadata: Mapping[str, Any],
) -> list[Any]:
    explicit = _metadata_value(metadata, f"{field}_order", f"{field}s")
    values: list[Any] = []
    values.extend(_metadata_order_values(explicit, field))
    observed: list[Any] = []
    for row in rows:
        if field == "workload":
            value: Any = _workload(row)
        elif field == "precision":
            value = _precision(row)
        elif field == "reuse_level":
            value = _reuse_level(row)
        elif field == "dataflow":
            value = _dataflow(row)
        elif field == "array_size":
            value = _int(row, "array_size", "array", "rows")
        else:
            value = _text(row, field)
        if value not in (None, "", "NA") and value not in observed:
            observed.append(value)
    for value in defaults:
        if value in observed and value not in values:
            values.append(value)
    for value in observed:
        if value not in values:
            values.append(value)
    return values


def _calibration_note(metadata: Mapping[str, Any]) -> str:
    value = _metadata_value(metadata, "calibration_status", "calibration", "technology_calibrated", "model_status", "status")
    if isinstance(value, Mapping):
        value = value.get("status") or value.get("label")
    if isinstance(value, bool):
        return "Analytical model; calibration status: calibrated (per experiment.json)." if value else "Analytical model; uncalibrated unless experiment.json provides calibration evidence."
    status = str(value or "").lower()
    if "calibrat" in status and "uncalibrat" not in status:
        return "Analytical model; calibration status: calibrated (per experiment.json)."
    return "Analytical model; uncalibrated unless experiment.json provides calibration evidence."


def _finish_figure(fig: plt.Figure, metadata: Mapping[str, Any]) -> None:
    fig.text(0.01, 0.006, _calibration_note(metadata), ha="left", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}.png", directory / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _selected_rows(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> list[dict[str, Any]]:
    """Collapse duplicate candidate rows to the lowest valid ADP per cell."""
    chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key_values: list[Any] = []
        for field in key_fields:
            if field == "workload":
                value: Any = _workload(row)
            elif field == "precision":
                value = _precision(row)
            elif field == "reuse_level":
                value = _reuse_level(row)
            elif field == "dataflow":
                value = _dataflow(row)
            elif field == "array_size":
                value = _int(row, "array_size", "array", "rows")
            else:
                value = _text(row, field)
            key_values.append(value)
        key = tuple(key_values)
        score = _float(row, "adp_mm2_cycles", "adp", "area_delay_product")
        current = chosen.get(key)
        current_score = _float(current, "adp_mm2_cycles", "adp", "area_delay_product") if current else math.nan
        if current is None or (math.isfinite(score) and (not math.isfinite(current_score) or score < current_score)):
            chosen[key] = dict(row)
    return list(chosen.values())


def _heatmap_matrix(
    rows: Sequence[Mapping[str, Any]],
    workloads: Sequence[str],
    precisions: Sequence[str],
    array_size: int,
    reuse: str,
    dataflow: str,
) -> np.ndarray:
    matrix = np.full((len(workloads), len(precisions)), np.nan)
    best: dict[tuple[str, str], tuple[float, str]] = {}
    for row in rows:
        if _int(row, "array_size", "array", "rows") != array_size:
            continue
        if _reuse_level(row) != reuse or _dataflow(row) != dataflow:
            continue
        workload, precision, primitive = _workload(row), _precision(row), _primitive(row)
        if workload not in workloads or precision not in precisions or primitive not in PRIMITIVE_ORDER:
            continue
        score = _float(row, "adp_mm2_cycles", "adp", "area_delay_product")
        # A selection row has one primitive, so the first row is sufficient if
        # ADP is not available.  If several tile candidates exist, keep best.
        key = (workload, precision)
        old = best.get(key)
        if old is None or (math.isfinite(score) and (not math.isfinite(old[0]) or score < old[0])):
            best[key] = (score, primitive)
    code = {name: float(index) for index, name in enumerate(PRIMITIVE_ORDER)}
    for (workload, precision), (_, primitive) in best.items():
        matrix[workloads.index(workload), precisions.index(precision)] = code[primitive]
    return matrix


def _draw_heatmap(ax: plt.Axes, matrix: np.ndarray, workloads: Sequence[str], precisions: Sequence[str], title: str) -> None:
    cmap = ListedColormap([PRIMITIVE_COLORS[name] for name in PRIMITIVE_ORDER]).with_extremes(bad="#eeeeee")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="none", aspect="auto")
    ax.set_xticks(range(len(precisions)), labels=precisions, rotation=35, ha="right")
    ax.set_yticks(range(len(workloads)), labels=workloads)
    ax.set_title(title, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = PRIMITIVE_ORDER[int(value)] if math.isfinite(value) else "NA"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")
    for edge in range(len(precisions) + 1):
        ax.axvline(edge - 0.5, color="white", linewidth=0.8)
    for edge in range(len(workloads) + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=0.8)


def _plot_heatmaps(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], directory: Path) -> list[Path]:
    raw_rows = list(rows)
    # Fixed rows retain every permitted square array/dataflow cell for the
    # requested facets. Tuned rows only retain the winning scale.
    rows = _scope_rows(raw_rows, "fixed")
    all_rows = rows
    workloads = _ordered_values(all_rows, "workload", [], metadata)
    precisions = _ordered_values(all_rows, "precision", DEFAULT_PRECISIONS, metadata)
    arrays = _ordered_values(all_rows, "array_size", DEFAULT_ARRAYS, metadata)
    arrays = [int(value) for value in arrays if value is not None]
    arrays = arrays or DEFAULT_ARRAYS
    if not workloads:
        workloads = ["NA"]
    if not precisions:
        precisions = DEFAULT_PRECISIONS
    dataflow = "weight_stationary"
    reuse = "high"
    outputs: list[Path] = []

    fig, axes = plt.subplots(1, len(arrays), figsize=(4.2 * len(arrays), max(3.6, 0.48 * len(workloads) + 1.8)), squeeze=False)
    for ax, array_size in zip(axes.ravel(), arrays):
        matrix = _heatmap_matrix(all_rows, workloads, precisions, array_size, reuse, dataflow)
        _draw_heatmap(ax, matrix, workloads, precisions, f"{array_size}×{array_size}")
    handles = [Line2D([0], [0], marker="s", linestyle="", color=PRIMITIVE_COLORS[name], label=name, markersize=9) for name in PRIMITIVE_ORDER]
    handles.append(Line2D([0], [0], marker="s", linestyle="", color="#eeeeee", markeredgecolor="#888888", label="NA", markersize=9))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle("Primitive selection heatmap · high reuse · weight-stationary", fontsize=13, y=1.02)
    _finish_figure(fig, metadata)
    outputs += _save(fig, directory, "fig1_primitive_heatmap")

    # Full facet for each array: reuse levels × dataflows.  Keeping one file
    # per array makes all workload rows visible while remaining printable.
    for array_size in arrays:
        fig, axes = plt.subplots(3, 2, figsize=(10, max(7, 0.32 * len(workloads) + 5)), squeeze=False)
        for i, reuse_level in enumerate(DEFAULT_REUSE_LEVELS):
            for j, flow in enumerate(DEFAULT_DATAFLOWS):
                matrix = _heatmap_matrix(all_rows, workloads, precisions, array_size, reuse_level, flow)
                _draw_heatmap(axes[i, j], matrix, workloads, precisions, f"{reuse_level} · {flow.replace('_', '-')}")
        fig.suptitle(f"Primitive selection facets · array {array_size}×{array_size}", fontsize=13, y=1.01)
        _finish_figure(fig, metadata)
        outputs += _save(fig, directory, f"fig1_heatmap_array{array_size}")
    return outputs


def _nondominated(points: Sequence[tuple[float, float, Mapping[str, Any]]]) -> list[tuple[float, float, Mapping[str, Any]]]:
    valid = [(a, l, row) for a, l, row in points if math.isfinite(a) and math.isfinite(l) and a > 0 and l > 0]
    result: list[tuple[float, float, Mapping[str, Any]]] = []
    for area, latency, row in valid:
        dominated = any(
            (other_area <= area and other_latency <= latency and (other_area < area or other_latency < latency))
            for other_area, other_latency, _ in valid
        )
        if not dominated:
            result.append((area, latency, row))
    return sorted(result, key=lambda item: (item[0], item[1]))


def _plot_pareto(rows: Sequence[Mapping[str, Any]], selections: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], directory: Path) -> list[Path]:
    source = _scope_rows(rows, "fixed") or _scope_rows(selections, "fixed")
    workloads = _ordered_values(source, "workload", [], metadata)
    precisions = _ordered_values(source, "precision", DEFAULT_PRECISIONS, metadata)
    target = next((name for name in workloads if _norm(name) == "attention"), workloads[0] if workloads else "NA")
    source = [row for row in source if _workload(row) == target and _reuse_level(row) == "high" and _dataflow(row) == "weight_stationary"]
    groups: dict[tuple[str, str], list[tuple[float, float, Mapping[str, Any]]]] = defaultdict(list)
    for row in source:
        primitive = _primitive(row)
        precision = _precision(row)
        if primitive not in PRIMITIVE_ORDER or precision not in precisions:
            continue
        area = _float(row, "area_mm2", "area")
        latency = _float(row, "latency_cycles", "latency")
        if math.isfinite(area) and math.isfinite(latency):
            groups[(primitive, precision)].append((area, latency, row))

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    any_point = False
    for (primitive, precision), points in sorted(groups.items()):
        front = _nondominated(points)
        if not front:
            continue
        any_point = True
        color = PRIMITIVE_COLORS[primitive]
        marker = PRECISION_MARKERS.get(precision, "o")
        ax.scatter([a for a, _, _ in points], [l for _, l, _ in points], color=color, marker=marker, alpha=0.45, s=40)
        # Grouping by precision is intentional: no line joins points from a
        # different precision, workload, or primitive.
        ax.plot([a for a, _, _ in front], [l for _, l, _ in front], color=color, linewidth=1.7, alpha=0.9)
    if not any_point:
        ax.text(0.5, 0.5, "NA: no feasible fixed Pareto points", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
    primitive_handles = [Line2D([0], [0], color=PRIMITIVE_COLORS[name], marker="o", linestyle="-", label=name) for name in PRIMITIVE_ORDER]
    marker_handles = [Line2D([0], [0], color="#555555", marker=PRECISION_MARKERS.get(p, "o"), linestyle="", label=p) for p in precisions]
    ax.legend(handles=primitive_handles + marker_handles, ncol=2, frameon=False, fontsize=8)
    ax.set_xlabel("Area (mm²)")
    ax.set_ylabel("Latency (cycles)")
    ax.set_title(f"Area–latency Pareto front · {target} · high reuse · weight-stationary")
    ax.grid(True, which="both", alpha=0.22)
    _finish_figure(fig, metadata)
    return _save(fig, directory, "fig2_pareto_area_latency")


def _comparison_ratio(row: Mapping[str, Any], direction: str) -> float:
    """Return MAC/LUT for reuse curves or LUT/MAC for scale curves."""
    mac_over_lut = _float(row, "mac_over_lut_adp")
    lut_over_mac = _float(row, "lut_over_mac_adp")
    mac_adp = _float(row, "mac_adp", "mac_adp_mm2_cycles")
    lut_adp = _float(row, "lut_adp", "lut_adp_mm2_cycles")
    if math.isfinite(mac_over_lut) and mac_over_lut > 0:
        mac_lut = mac_over_lut
    elif math.isfinite(lut_over_mac) and lut_over_mac > 0:
        mac_lut = 1.0 / lut_over_mac
    elif math.isfinite(mac_adp) and math.isfinite(lut_adp) and mac_adp > 0 and lut_adp > 0:
        mac_lut = mac_adp / lut_adp
    else:
        return math.nan
    return mac_lut if direction == "mac_over_lut" else (1.0 / mac_lut if mac_lut > 0 else math.nan)


def _plot_segments(ax: plt.Axes, x_values: Sequence[float], y_values: Sequence[float], **kwargs: Any) -> None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    start: int | None = None
    for index, is_valid in enumerate(finite):
        if is_valid and start is None:
            start = index
        if (not is_valid or index == len(finite) - 1) and start is not None:
            end = index if is_valid and index == len(finite) - 1 else index - 1
            if end - start >= 0:
                ax.plot(x[start : end + 1], y[start : end + 1], **kwargs)
            start = None


def _plot_reuse(comparisons: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = _scope_rows(comparisons, "fixed")
    workloads = _ordered_values(rows, "workload", [], metadata)
    precisions = _ordered_values(rows, "precision", DEFAULT_PRECISIONS, metadata)
    flows = [flow for flow in DEFAULT_DATAFLOWS if any(_dataflow(row) == flow for row in rows)] or _ordered_values(rows, "dataflow", DEFAULT_DATAFLOWS, metadata)
    workloads = workloads or ["NA"]
    precisions = precisions or list(DEFAULT_PRECISIONS)
    flows = flows or list(DEFAULT_DATAFLOWS)
    outputs: list[Path] = []
    for precision in precisions:
        subset = [row for row in rows if _precision(row) == precision and _int(row, "array_size", "array", "rows") == 16]
        expected_x = sorted({x for x in (_float(row, "reuse_factor", "selected_reuse_factor", "reuse_multiplier") for row in subset) if math.isfinite(x) and x > 0})
        fig, axes = plt.subplots(len(workloads), len(flows), figsize=(4.7 * len(flows), max(2.1 * len(workloads), 6)), squeeze=False)
        for wi, workload in enumerate(workloads):
            for fi, flow in enumerate(flows):
                ax = axes[wi, fi]
                panel = [row for row in subset if _workload(row) == workload and _dataflow(row) == flow]
                by_x: dict[float, float] = {}
                for row in panel:
                    x = _float(row, "reuse_factor", "selected_reuse_factor", "reuse_multiplier")
                    y = _comparison_ratio(row, "mac_over_lut")
                    if math.isfinite(x) and x > 0 and math.isfinite(y) and y > 0:
                        by_x[x] = y
                ys = [by_x.get(x, math.nan) for x in expected_x]
                if any(math.isfinite(y) for y in ys):
                    _plot_segments(ax, expected_x, ys, marker="o", color="#4c78a8", linewidth=1.4, markersize=4)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                else:
                    ax.text(0.5, 0.5, "NA", ha="center", va="center", transform=ax.transAxes)
                ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
                ax.set_title(f"{workload} · {flow.replace('_', '-')}", fontsize=9)
                ax.grid(True, which="both", alpha=0.18)
                if wi == len(workloads) - 1:
                    ax.set_xlabel("Reuse factor")
                if fi == 0:
                    ax.set_ylabel("MAC ADP / LUT ADP")
        fig.suptitle(f"Reuse crossover · {precision} · array 16×16", fontsize=13, y=1.01)
        _finish_figure(fig, metadata)
        outputs += _save(fig, directory, f"fig3_reuse_crossover_{precision}")
    return outputs


def _plot_scale(comparisons: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = _scope_rows(comparisons, "fixed")
    workloads = _ordered_values(rows, "workload", [], metadata) or ["NA"]
    precisions = _ordered_values(rows, "precision", DEFAULT_PRECISIONS, metadata)
    flows = [flow for flow in DEFAULT_DATAFLOWS if any(_dataflow(row) == flow for row in rows)] or _ordered_values(rows, "dataflow", DEFAULT_DATAFLOWS, metadata)
    precisions = precisions or list(DEFAULT_PRECISIONS)
    flows = flows or list(DEFAULT_DATAFLOWS)
    arrays = sorted({a for a in (_int(row, "array_size", "array", "rows") for row in rows) if a is not None}) or DEFAULT_ARRAYS
    outputs: list[Path] = []
    for workload in workloads:
        fig, axes = plt.subplots(len(precisions), len(flows), figsize=(4.8 * len(flows), max(2.5 * len(precisions), 7)), squeeze=False)
        for pi, precision in enumerate(precisions):
            for fi, flow in enumerate(flows):
                ax = axes[pi, fi]
                panel = [row for row in rows if _workload(row) == workload and _precision(row) == precision and _dataflow(row) == flow and _reuse_level(row) == "high"]
                by_array: dict[int, float] = {}
                for row in panel:
                    array_size = _int(row, "array_size", "array", "rows")
                    ratio = _comparison_ratio(row, "lut_over_mac")
                    if array_size is not None and math.isfinite(ratio) and ratio > 0:
                        by_array[array_size] = ratio
                ys = [by_array.get(array_size, math.nan) for array_size in arrays]
                if any(math.isfinite(y) for y in ys):
                    _plot_segments(ax, arrays, ys, marker="o", color="#984ea3", linewidth=1.5, markersize=4)
                    ax.set_yscale("log")
                else:
                    ax.text(0.5, 0.5, "NA", ha="center", va="center", transform=ax.transAxes)
                ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
                ax.set_xticks(arrays)
                ax.set_title(f"{precision} · {flow.replace('_', '-')}", fontsize=9)
                ax.grid(True, which="both", alpha=0.18)
                if pi == len(precisions) - 1:
                    ax.set_xlabel("Array size (N×N)")
                if fi == 0:
                    ax.set_ylabel("LUT ADP / MAC ADP")
        fig.suptitle(f"Array-scale crossover · {workload} · high reuse", fontsize=13, y=1.01)
        _finish_figure(fig, metadata)
        safe_workload = re.sub(r"[^A-Za-z0-9_-]+", "_", workload)
        outputs += _save(fig, directory, f"fig4_scale_crossover_{safe_workload}")
    return outputs


def plot_results(directory: str | Path) -> list[Path]:
    """Generate all Experiment 2 figures and return written PNG/PDF paths."""
    directory = Path(directory)
    metadata = _read_metadata(directory)
    selections = _read_csv(directory / "selections.csv")
    comparisons = _read_csv(directory / "comparisons.csv")
    pareto = _read_csv(directory / "pareto.csv")
    outputs: list[Path] = []
    outputs += _plot_heatmaps(selections, metadata, directory)
    outputs += _plot_pareto(pareto, selections, metadata, directory)
    outputs += _plot_reuse(comparisons, metadata, directory)
    outputs += _plot_scale(comparisons, metadata, directory)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="experiments/exp2/results", help="directory containing selections.csv, comparisons.csv, pareto.csv")
    args = parser.parse_args()
    outputs = plot_results(args.input)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

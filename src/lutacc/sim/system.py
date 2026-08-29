"""Decoder-layer aggregation over the shared compiler/simulator interfaces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..schema import GemmShape, ModelSpec
from .baselines import (
    FIGLUTCycleModel,
    GPURooflineModel,
    simulate_figlut_gemm,
    simulate_gpu_roofline_gemm,
)
from .oasis import OasisCycleModel, simulate_oasis_gemm


@dataclass(frozen=True)
class DecoderSystemResult:
    design: str
    model: str
    batch: int
    cycles_per_token: int
    latency_ms: float
    tokens_per_second: float
    energy_per_token_mj: float
    operations_per_token: int
    dram_bytes_per_token: int
    layer_breakdown: Tuple[Dict[str, Any], ...]
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["layer_breakdown"] = list(self.layer_breakdown)
        return result


def decoder_linear_shapes(model: ModelSpec, batch: int = 1) -> List[Tuple[str, GemmShape]]:
    return [
        (
            "qkv",
            GemmShape(
                1,
                model.hidden_size,
                model.hidden_size + 2 * model.kv_heads * model.head_dim,
                batch,
            ),
        ),
        ("o_proj", GemmShape(1, model.hidden_size, model.hidden_size, batch)),
        ("gate_up", GemmShape(1, model.hidden_size, 2 * model.intermediate_size, batch)),
        ("down", GemmShape(1, model.intermediate_size, model.hidden_size, batch)),
    ]


def simulate_decoder_linear(
    model: ModelSpec,
    design: str,
    activation_bits: int = 4,
    batch: int = 1,
    oasis_lookahead: bool = True,
) -> DecoderSystemResult:
    if design not in {"oasis", "figlut", "a100_fp16", "quarot_w4a4"}:
        raise ValueError("unsupported decoder design")
    layer_results = []
    for name, shape in decoder_linear_shapes(model, batch):
        if design == "oasis":
            result = simulate_oasis_gemm(
                shape,
                activation_bits,
                0.01,
                resident_weights=False,
                lookahead=oasis_lookahead,
            )
        elif design == "figlut":
            result = simulate_figlut_gemm(shape, resident_weights=False)
        else:
            result = simulate_gpu_roofline_gemm(shape, design)
        layer_results.append((name, result))
    clock_hz = layer_results[0][1].clock_hz
    cycles_per_layer = sum(result.cycles for _, result in layer_results)
    cycles_per_token = cycles_per_layer * model.layers
    latency_s = cycles_per_token / clock_hz
    # Paper-normalized system energy uses each compared design's nominal active
    # power over its end-to-end cycle count. Per-stage/DRAM energy remains in the
    # underlying GEMM result for breakdown studies, but mixing those calibrated
    # stage coefficients with FIGLUT's published whole-engine power would not be
    # a like-for-like comparison.
    if design == "oasis":
        nominal_power_w = OasisCycleModel.from_files().power_w
    elif design == "figlut":
        nominal_power_w = FIGLUTCycleModel.from_file().power_w
    else:
        nominal_power_w = GPURooflineModel.from_file(design).active_power_w
    energy_per_token = nominal_power_w * cycles_per_token / clock_hz * 1000.0
    operations = sum(result.operations for _, result in layer_results) * model.layers
    dram_bytes = sum(
        result.dram_read_bytes + result.dram_write_bytes for _, result in layer_results
    ) * model.layers
    breakdown = tuple(
        {
            "operation": name,
            "shape": shape.to_dict(),
            "cycles": result.cycles,
            "energy_mj": result.energy_mj,
        }
        for (name, shape), (_, result) in zip(decoder_linear_shapes(model, batch), layer_results)
    )
    return DecoderSystemResult(
        design=(
            ("oasis" if oasis_lookahead else "oasis-c") + "-a%d" % activation_bits
            if design == "oasis"
            else "figlut-q4" if design == "figlut" else design.replace("_", "-")
        ),
        model=model.name,
        batch=batch,
        cycles_per_token=cycles_per_token,
        latency_ms=latency_s * 1000.0,
        tokens_per_second=batch / latency_s,
        energy_per_token_mj=energy_per_token,
        operations_per_token=operations,
        dram_bytes_per_token=dram_bytes,
        layer_breakdown=breakdown,
        evidence=(
            "predicted"
            if design == "oasis"
            else "inferred" if design == "figlut" else "calibrated_roofline"
        ),
    )


def compare_oasis_lookahead(model: ModelSpec, activation_bits: int) -> Dict[str, Any]:
    lookahead = simulate_decoder_linear(
        model, "oasis", activation_bits, 1, oasis_lookahead=True
    )
    conventional = simulate_decoder_linear(
        model, "oasis", activation_bits, 1, oasis_lookahead=False
    )
    return {
        "model": model.name,
        "activation_bits": activation_bits,
        "lookahead": lookahead.to_dict(),
        "conventional_dynamic_detection": conventional.to_dict(),
        "throughput_improvement": (
            lookahead.tokens_per_second / conventional.tokens_per_second
        ),
        "comparison_evidence": "predicted_mechanism_ablation",
    }


def geometric_mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items or any(value <= 0 for value in items):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in items) / len(items))


def compare_oasis_figlut_decode(
    models: Sequence[ModelSpec], activation_bits: int = 4, batch: int = 1
) -> Dict[str, Any]:
    points = []
    for model in models:
        oasis = simulate_decoder_linear(model, "oasis", activation_bits, batch)
        figlut = simulate_decoder_linear(model, "figlut", 4, batch)
        points.append(
            {
                "model": model.name,
                "oasis": oasis.to_dict(),
                "figlut": figlut.to_dict(),
                "speedup": oasis.tokens_per_second / figlut.tokens_per_second,
                "energy_efficiency_improvement": (
                    figlut.energy_per_token_mj / oasis.energy_per_token_mj
                ),
            }
        )
    return {
        "activation_bits": activation_bits,
        "batch": batch,
        "points": points,
        "geomean_speedup": geometric_mean(point["speedup"] for point in points),
        "geomean_energy_efficiency_improvement": geometric_mean(
            point["energy_efficiency_improvement"] for point in points
        ),
        "comparison_evidence": "predicted_vs_inferred",
    }


def compare_oasis_gpu_decode(
    models: Sequence[ModelSpec], activation_bits: int = 4, batch: int = 1
) -> Dict[str, Any]:
    points = []
    for model in models:
        oasis = simulate_decoder_linear(model, "oasis", activation_bits, batch)
        baselines = {
            name: simulate_decoder_linear(model, name, 4, batch)
            for name in ("a100_fp16", "quarot_w4a4")
        }
        points.append(
            {
                "model": model.name,
                "oasis": oasis.to_dict(),
                "baselines": {
                    name: result.to_dict() for name, result in baselines.items()
                },
                "comparisons": {
                    name: {
                        "speedup": oasis.tokens_per_second / result.tokens_per_second,
                        "energy_efficiency_improvement": (
                            result.energy_per_token_mj / oasis.energy_per_token_mj
                        ),
                    }
                    for name, result in baselines.items()
                },
            }
        )
    geomeans = {}
    for name in ("a100_fp16", "quarot_w4a4"):
        geomeans[name] = {
            "speedup": geometric_mean(
                point["comparisons"][name]["speedup"] for point in points
            ),
            "energy_efficiency_improvement": geometric_mean(
                point["comparisons"][name]["energy_efficiency_improvement"]
                for point in points
            ),
        }
    return {
        "activation_bits": activation_bits,
        "batch": batch,
        "points": points,
        "geomeans": geomeans,
        "comparison_evidence": "predicted_vs_calibrated_roofline_not_native",
    }

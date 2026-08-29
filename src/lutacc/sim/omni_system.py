"""Long-context equal-throughput Omni-LUT versus FIGLUT energy proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from ..config import load_yaml, repository_root
from ..schema import AttentionShape, GemmShape, ModelSpec
from .omni import simulate_omni_attention, simulate_omni_gemm


@dataclass(frozen=True)
class OmniTransformerResult:
    model: str
    phase: str
    batch: int
    query_length: int
    kv_length: int
    cycles: int
    latency_ms: float
    tokens_per_second: float
    energy_mj: float
    operations: int
    dram_bytes: int
    layer_breakdown: Tuple[Dict[str, Any], ...]
    evidence: str = "predicted_from_inferred_configuration"

    def to_dict(self) -> Dict[str, Any]:
        document = asdict(self)
        document["layer_breakdown"] = list(self.layer_breakdown)
        return document


def simulate_omni_transformer(
    model: ModelSpec,
    phase: str,
    query_length: int,
    kv_length: int,
    batch: int = 1,
    average_key_bits: float = 3.0,
    average_value_bits: float = 3.0,
) -> OmniTransformerResult:
    """Aggregate all four AW-GEMMs and both AA-GEMMs for a transformer phase."""

    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be prefill or decode")
    if min(batch, query_length, kv_length) <= 0:
        raise ValueError("transformer workload dimensions must be positive")
    if phase == "decode" and query_length != 1:
        raise ValueError("decode phase models one newly generated token")
    linear_shapes = (
        (
            "qkv",
            GemmShape(
                query_length,
                model.hidden_size,
                model.hidden_size + 2 * model.kv_heads * model.head_dim,
                batch,
            ),
        ),
        ("o_proj", GemmShape(query_length, model.hidden_size, model.hidden_size, batch)),
        (
            "gate_up",
            GemmShape(
                query_length,
                model.hidden_size,
                2 * model.intermediate_size,
                batch,
            ),
        ),
        (
            "down",
            GemmShape(
                query_length,
                model.intermediate_size,
                model.hidden_size,
                batch,
            ),
        ),
    )
    qkv = simulate_omni_gemm(linear_shapes[0][1], resident_weights=False)
    attention = simulate_omni_attention(
        AttentionShape(
            batch,
            model.attention_heads,
            query_length,
            kv_length,
            model.head_dim,
        ),
        phase,
        average_key_bits,
        average_value_bits,
    )
    remaining = [
        (name, simulate_omni_gemm(shape, resident_weights=False))
        for name, shape in linear_shapes[1:]
    ]
    per_layer = [("qkv", qkv), ("attention_qk_softmax_pv", attention)] + remaining
    cycles_per_layer = sum(result.cycles for _, result in per_layer)
    cycles = cycles_per_layer * model.layers
    clock_hz = per_layer[0][1].clock_hz
    latency_s = cycles / clock_hz
    return OmniTransformerResult(
        model=model.name,
        phase=phase,
        batch=batch,
        query_length=query_length,
        kv_length=kv_length,
        cycles=cycles,
        latency_ms=latency_s * 1000.0,
        tokens_per_second=batch * query_length / latency_s,
        energy_mj=sum(result.energy_mj for _, result in per_layer) * model.layers,
        operations=sum(result.operations for _, result in per_layer) * model.layers,
        dram_bytes=sum(
            result.dram_read_bytes + result.dram_write_bytes
            for _, result in per_layer
        )
        * model.layers,
        layer_breakdown=tuple(
            {
                "operation": name,
                "cycles": result.cycles,
                "energy_mj": result.energy_mj,
                "design": result.design,
                "workload": result.workload,
            }
            for name, result in per_layer
        ),
    )


def _linear_work(model: ModelSpec) -> Dict[str, float]:
    qkv_n = model.hidden_size + 2 * model.kv_heads * model.head_dim
    weight_elements_per_layer = (
        model.hidden_size * qkv_n
        + model.hidden_size * model.hidden_size
        + model.hidden_size * (2 * model.intermediate_size)
        + model.intermediate_size * model.hidden_size
    )
    return {
        "operations": float(2 * model.layers * weight_elements_per_layer),
        "packed_w4_bytes": float(model.layers * weight_elements_per_layer * 4 / 8),
    }


def compare_omni_figlut_long_context(
    model: ModelSpec,
    contexts: Optional[Sequence[int]] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    path = config_path or repository_root() / "configs" / "calibration" / "omni_energy_proxy_v1.yaml"
    config = load_yaml(str(path))
    selected = list(contexts or config["contexts"])
    if not selected or min(selected) <= 0:
        raise ValueError("contexts must be positive")
    linear = _linear_work(model)
    points = []
    for context in selected:
        # QK^T and PV each contain one MAC per head/context/head_dim.
        aa_operations = float(
            4 * model.layers * model.attention_heads * context * model.head_dim
        )
        kv_elements = float(
            model.layers * context * model.kv_heads * model.head_dim * 2
        )
        figlut_kv_bytes = kv_elements * 2
        omni_kv_bytes = kv_elements * (config["key_bits"] + config["value_bits"]) / 16
        common_weight_energy = linear["packed_w4_bytes"] * config["dram_pj_per_byte"]
        figlut_energy_pj = (
            linear["operations"] * config["figlut_aw_pj_per_op"]
            + aa_operations * config["fp16_aa_pj_per_op"]
            + common_weight_energy
            + figlut_kv_bytes * config["dram_pj_per_byte"]
        )
        omni_energy_pj = (
            linear["operations"] * config["omni_aw_pj_per_op"]
            + aa_operations * config["omni_lut_aa_pj_per_op"]
            + common_weight_energy
            + omni_kv_bytes * config["dram_pj_per_byte"]
        )
        points.append(
            {
                "context": context,
                "figlut_energy_mj": figlut_energy_pj / 1e9,
                "omni_energy_mj": omni_energy_pj / 1e9,
                "energy_efficiency_improvement": figlut_energy_pj / omni_energy_pj,
                "aa_operations": int(aa_operations),
                "figlut_kv_bytes": int(figlut_kv_bytes),
                "omni_kv_bytes": int(omni_kv_bytes),
            }
        )
    improvements = [point["energy_efficiency_improvement"] for point in points]
    return {
        "model": model.name,
        "points": points,
        "minimum_improvement": min(improvements),
        "maximum_improvement": max(improvements),
        "configuration": config,
        "evidence": "predicted_from_inferred_28nm_coefficients",
    }

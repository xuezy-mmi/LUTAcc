"""Validated bridge from compiler IR to mechanism-level cycle simulators."""

from __future__ import annotations

from typing import Optional, Set

from ..compiler.ir import Program
from ..schema import AttentionShape, GemmShape, SimulationResult
from .oasis import OasisCycleModel, simulate_oasis_gemm
from .omni import OmniCycleModel, simulate_omni_attention, simulate_omni_gemm


OASIS_KINDS: Set[str] = {
    "dma_read",
    "nearest_centroid",
    "orizuru_top_bottom_k",
    "cartesian_index_concat",
    "parallel_histogram",
    "cartesian_lut_weighted_sum",
    "outlier_residual_gemm",
    "vector_add",
}
OMNI_KINDS: Set[str] = {
    "as_bit_allocate",
    "lgu_compensated_lut",
    "activation_activation_lut_gemm",
    "softmax",
}
OMNI_AW_KINDS: Set[str] = {
    "dma_read",
    "lgu_weight_lut",
    "activation_weight_lut_gemm",
}


def _require_mechanisms(program: Program, expected: Set[str]) -> None:
    actual = {operation.kind for operation in program.operations}
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            "IR mechanism mismatch: missing=%s unknown=%s"
            % (sorted(missing), sorted(unknown))
        )


def simulate_program(
    program: Program,
    *,
    resident_weights: bool = True,
    oasis_model: Optional[OasisCycleModel] = None,
    omni_model: Optional[OmniCycleModel] = None,
) -> SimulationResult:
    """Simulate a validated compiler program without rebuilding its workload.

    The bridge deliberately accepts only the two canonical lowering vocabularies.
    This prevents a modified or stale IR file from silently being simulated as a
    different mechanism based solely on a convenient shape field.
    """

    program.validate()
    shape_document = program.metadata.get("shape")
    if not isinstance(shape_document, dict):
        raise ValueError("IR metadata.shape is required for cycle simulation")
    if program.design == "oasis":
        _require_mechanisms(program, OASIS_KINDS)
        quantize = next(
            operation
            for operation in program.operations
            if operation.kind == "nearest_centroid"
        )
        result = simulate_oasis_gemm(
            GemmShape(**shape_document),
            activation_bits=int(quantize.attributes["activation_bits"]),
            outlier_fraction=float(quantize.attributes["outlier_fraction"]),
            model=oasis_model,
            resident_weights=resident_weights,
        )
    elif program.design == "omni-lut":
        if program.metadata.get("workload_kind") == "aw_gemm":
            _require_mechanisms(program, OMNI_AW_KINDS)
            lookup = next(
                operation
                for operation in program.operations
                if operation.kind == "activation_weight_lut_gemm"
            )
            result = simulate_omni_gemm(
                GemmShape(**shape_document),
                weight_bits=int(lookup.attributes["weight_bits"]),
                model=omni_model,
                resident_weights=resident_weights,
            )
        else:
            _require_mechanisms(program, OMNI_KINDS)
            allocation = next(
                operation
                for operation in program.operations
                if operation.kind == "as_bit_allocate"
            )
            phase = str(allocation.attributes["phase"])
            if program.metadata.get("phase_adaptive_dataflow") != (
                "prefill_ws" if phase == "prefill" else "decode_os_v"
            ):
                raise ValueError("IR phase and phase-adaptive dataflow disagree")
            result = simulate_omni_attention(
                AttentionShape(**shape_document),
                phase=phase,
                average_key_bits=float(allocation.attributes["average_k_bits"]),
                average_value_bits=float(allocation.attributes["average_v_bits"]),
                model=omni_model,
            )
    else:  # Program.validate normally catches this, retained for type narrowing.
        raise ValueError("unsupported IR design %s" % program.design)
    result.metadata.update(
        {
            "ir_program": program.name,
            "ir_schema_version": program.schema_version,
            "compiler_simulator_contract": "validated",
        }
    )
    return result

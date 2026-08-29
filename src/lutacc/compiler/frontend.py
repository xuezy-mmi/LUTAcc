"""Shape frontends that lower GEMM and attention into LUTAcc IR."""

from __future__ import annotations

from typing import Any, Dict, List

from ..schema import AttentionShape, GemmShape, ModelSpec
from .ir import Operation, Program, TensorSpec, tensor


def compile_oasis_gemm(
    shape: GemmShape,
    weight_bits: int = 4,
    activation_bits: int = 4,
    outlier_fraction: float = 0.01,
) -> Program:
    if weight_bits != 4 or activation_bits not in (3, 4):
        raise ValueError("OASIS reconstruction supports W4A3 and W4A4")
    if not 0.0 <= outlier_fraction < 1.0:
        raise ValueError("outlier_fraction must be in [0, 1)")
    m_total = shape.batch * shape.m
    tensors = [
        tensor("activation_fp16", (m_total, shape.k), "fp16", "dram"),
        tensor("weight_index", (shape.k, shape.n), "index4", "dram"),
        tensor("activation_stream", (m_total, shape.k), "fp16", "stream"),
        tensor(
            "activation_index",
            (m_total, shape.k),
            "index%d" % activation_bits,
            "sram",
        ),
        tensor("outlier_pairs", (m_total, max(1, int(shape.k * outlier_fraction))), "fp16", "stream"),
        tensor("concat_index", (m_total, shape.k, shape.n), "uint8", "stream"),
        tensor(
            "index_histogram",
            (m_total, shape.n, 1 << (weight_bits + activation_bits)),
            "fp16",
            "register",
            semantic_dtype="integer_count",
        ),
        tensor("lookahead_output", (m_total, shape.n), "fp16", "sram"),
        tensor("compensation", (m_total, shape.n), "fp16", "sram"),
        tensor("output_fp16", (m_total, shape.n), "fp16", "dram"),
    ]
    common = {
        "m": m_total,
        "k": shape.k,
        "n": shape.n,
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "outlier_fraction": outlier_fraction,
    }
    operations = [
        Operation(
            "load_activation",
            "dma_read",
            ("activation_fp16",),
            ("activation_stream",),
            resource="dram_read",
            attributes={"streaming": True},
        ),
        Operation(
            "activation_quantize",
            "nearest_centroid",
            ("activation_stream",),
            ("activation_index",),
            ("load_activation",),
            "clustering_unit",
            {**common, "streaming": True},
        ),
        Operation(
            "detect_outliers",
            "orizuru_top_bottom_k",
            ("activation_stream",),
            ("outlier_pairs",),
            ("load_activation",),
            "orizuru",
            {**common, "streaming": True, "tie_break": "lowest_index"},
        ),
        Operation(
            "concatenate_indices",
            "cartesian_index_concat",
            ("activation_index", "weight_index"),
            ("concat_index",),
            ("activation_quantize",),
            "concat_units",
            {**common, "streaming_dependency": True},
        ),
        Operation(
            "count_indices",
            "parallel_histogram",
            ("concat_index",),
            ("index_histogram",),
            ("concatenate_indices",),
            "index_counters",
            {**common, "streaming_dependency": True},
        ),
        Operation(
            "lut_reduce",
            "cartesian_lut_weighted_sum",
            ("index_histogram",),
            ("lookahead_output",),
            ("count_indices",),
            "mac_tree",
            {**common, "streaming_dependency": True},
        ),
        Operation(
            "error_compensate",
            "outlier_residual_gemm",
            ("outlier_pairs", "weight_index"),
            ("compensation",),
            ("detect_outliers",),
            "compensation_macs",
            common,
        ),
        Operation(
            "merge_branches",
            "vector_add",
            ("lookahead_output", "compensation"),
            ("output_fp16",),
            ("lut_reduce", "error_compensate"),
            "merge_macs",
            common,
        ),
    ]
    program = Program(
        name="oasis-gemm-%dx%dx%d" % (m_total, shape.k, shape.n),
        design="oasis",
        tensors=tensors,
        operations=operations,
        metadata={
            "shape": shape.to_dict(),
            "mechanism": "precomputed_cartesian_product_lut",
            "branch_overlap": True,
        },
    )
    program.validate()
    return program


def compile_omni_attention(
    shape: AttentionShape,
    phase: str,
    average_k_bits: float = 3.0,
    average_v_bits: float = 3.0,
) -> Program:
    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be prefill or decode")
    if not 2.0 <= average_k_bits <= 4.0 or not 2.0 <= average_v_bits <= 4.0:
        raise ValueError("Omni-LUT AS-Bit budget must be in [2, 4]")
    prefix = (shape.batch, shape.heads)
    tensors = [
        tensor("query_fp16", prefix + (shape.query_length, shape.head_dim), "fp16", "dram"),
        tensor("key_fp16", prefix + (shape.kv_length, shape.head_dim), "fp16", "dram"),
        tensor("value_fp16", prefix + (shape.kv_length, shape.head_dim), "fp16", "dram"),
        tensor("key_index", prefix + (shape.kv_length, shape.head_dim), "uint4", "sram", variable_bits=True),
        tensor("value_index", prefix + (shape.kv_length, shape.head_dim), "uint4", "sram", variable_bits=True),
        tensor("key_lut", prefix + (shape.query_length, 16), "fp16", "sram"),
        tensor("score_fp16", prefix + (shape.query_length, shape.kv_length), "fp16", "sram"),
        tensor("probability_fp16", prefix + (shape.query_length, shape.kv_length), "fp16", "sram"),
        tensor("value_lut", prefix + (shape.query_length, 16), "fp16", "sram"),
        tensor("attention_output", prefix + (shape.query_length, shape.head_dim), "fp16", "dram"),
    ]
    attrs: Dict[str, Any] = {
        **shape.to_dict(),
        "phase": phase,
        "dataflow": "prefill_ws" if phase == "prefill" else "decode_os_v",
        "average_k_bits": average_k_bits,
        "average_v_bits": average_v_bits,
    }
    operations = [
        Operation(
            "allocate_k_bits",
            "as_bit_allocate",
            ("key_fp16",),
            ("key_index",),
            resource="compiler_metadata",
            attributes={**attrs, "direction": "per_channel"},
        ),
        Operation(
            "allocate_v_bits",
            "as_bit_allocate",
            ("value_fp16",),
            ("value_index",),
            resource="compiler_metadata",
            attributes={**attrs, "direction": "per_token"},
        ),
        Operation(
            "generate_qk_lut",
            "lgu_compensated_lut",
            ("query_fp16", "key_index"),
            ("key_lut",),
            ("allocate_k_bits",),
            "lgu",
            attrs,
        ),
        Operation(
            "qk_lut_gemm",
            "activation_activation_lut_gemm",
            ("query_fp16", "key_index", "key_lut"),
            ("score_fp16",),
            ("generate_qk_lut",),
            "lut_pe_array",
            attrs,
        ),
        Operation(
            "softmax",
            "softmax",
            ("score_fp16",),
            ("probability_fp16",),
            ("qk_lut_gemm",),
            "special_function",
            attrs,
        ),
        Operation(
            "generate_pv_lut",
            "lgu_compensated_lut",
            ("probability_fp16", "value_index"),
            ("value_lut",),
            ("softmax", "allocate_v_bits"),
            "lgu",
            attrs,
        ),
        Operation(
            "pv_lut_gemm",
            "activation_activation_lut_gemm",
            ("probability_fp16", "value_index", "value_lut"),
            ("attention_output",),
            ("generate_pv_lut",),
            "lut_pe_array",
            attrs,
        ),
    ]
    program = Program(
        name="omni-attention-%s-q%d-kv%d-d%d"
        % (phase, shape.query_length, shape.kv_length, shape.head_dim),
        design="omni-lut",
        tensors=tensors,
        operations=operations,
        metadata={
            "shape": shape.to_dict(),
            "workload_kind": "attention",
            "phase_adaptive_dataflow": attrs["dataflow"],
            "supports_aw_and_aa_gemm": True,
        },
    )
    program.validate()
    return program


def compile_omni_gemm(shape: GemmShape, weight_bits: int = 4) -> Program:
    """Lower Omni-LUT activation-weight GEMM to its LGU/LUT-PE path."""

    if weight_bits != 4:
        raise ValueError("the open Omni-LUT AW path supports W4 weights")
    rows = shape.batch * shape.m
    levels = 1 << weight_bits
    tensors = [
        tensor("activation_fp16", (rows, shape.k), "fp16", "dram"),
        tensor("weight_index", (shape.k, shape.n), "index4", "dram"),
        tensor("weight_centroid", (levels,), "fp16", "sram"),
        tensor("weight_scale", (shape.n,), "fp16", "sram"),
        tensor("activation_stream", (rows, shape.k), "fp16", "stream"),
        tensor("aw_product_lut", (rows, shape.k, levels), "fp16", "stream"),
        tensor("output_fp16", (rows, shape.n), "fp16", "dram"),
    ]
    attributes = {
        **shape.to_dict(),
        "weight_bits": weight_bits,
        "lut_levels": levels,
        "dataflow": "weight_stationary",
    }
    operations = [
        Operation(
            "load_activation",
            "dma_read",
            ("activation_fp16",),
            ("activation_stream",),
            resource="dram_read",
            attributes={**attributes, "streaming": True},
        ),
        Operation(
            "generate_aw_lut",
            "lgu_weight_lut",
            ("activation_stream", "weight_centroid"),
            ("aw_product_lut",),
            ("load_activation",),
            "lgu",
            {**attributes, "streaming_dependency": True},
        ),
        Operation(
            "aw_lut_gemm",
            "activation_weight_lut_gemm",
            ("aw_product_lut", "weight_index", "weight_scale"),
            ("output_fp16",),
            ("generate_aw_lut",),
            "lut_pe_array",
            attributes,
        ),
    ]
    program = Program(
        name="omni-aw-gemm-%dx%dx%d" % (rows, shape.k, shape.n),
        design="omni-lut",
        tensors=tensors,
        operations=operations,
        metadata={
            "shape": shape.to_dict(),
            "workload_kind": "aw_gemm",
            "supports_aw_and_aa_gemm": True,
            "configuration_evidence": "inferred",
        },
    )
    program.validate()
    return program


def compile_transformer_layer(
    model: ModelSpec,
    batch: int,
    prompt_length: int,
    decode_tokens: int,
    design: str,
) -> List[Program]:
    if min(batch, prompt_length) <= 0 or decode_tokens < 0:
        raise ValueError("invalid transformer workload")
    programs: List[Program] = []
    phases = [("prefill", prompt_length)]
    if decode_tokens:
        phases.append(("decode", 1))
    for phase, tokens in phases:
        if design == "omni-lut":
            qkv = compile_omni_gemm(
                GemmShape(
                    tokens,
                    model.hidden_size,
                    model.hidden_size + 2 * model.kv_heads * model.head_dim,
                    batch=batch,
                )
            )
            qkv.name = "%s-%s-qkv" % (model.name, phase)
            programs.append(qkv)
            attention = (
                compile_omni_attention(
                    AttentionShape(
                        batch,
                        model.attention_heads,
                        tokens,
                        prompt_length if phase == "prefill" else prompt_length + decode_tokens,
                        model.head_dim,
                    ),
                    phase,
                )
            )
            attention.name = "%s-%s-attention" % (model.name, phase)
            programs.append(attention)
            for suffix, input_dim, output_dim in (
                ("o_proj", model.hidden_size, model.hidden_size),
                ("gate_up", model.hidden_size, 2 * model.intermediate_size),
                ("down", model.intermediate_size, model.hidden_size),
            ):
                program = compile_omni_gemm(
                    GemmShape(tokens, input_dim, output_dim, batch=batch)
                )
                program.name = "%s-%s-%s" % (model.name, phase, suffix)
                programs.append(program)
            for program in programs[-5:]:
                program.metadata.update(
                    model=model.to_dict(), phase=phase, decode_tokens=decode_tokens
                )
        elif design == "oasis":
            for suffix, input_dim, output_dim in (
                ("qkv", model.hidden_size, model.hidden_size + 2 * model.kv_heads * model.head_dim),
                ("o_proj", model.hidden_size, model.hidden_size),
                ("gate_up", model.hidden_size, 2 * model.intermediate_size),
                ("down", model.intermediate_size, model.hidden_size),
            ):
                program = compile_oasis_gemm(
                    GemmShape(tokens, input_dim, output_dim, batch=batch)
                )
                program.name = "%s-%s-%s" % (model.name, phase, suffix)
                program.metadata.update(
                    model=model.to_dict(), phase=phase, decode_tokens=decode_tokens
                )
                programs.append(program)
        else:
            raise ValueError("design must be oasis or omni-lut")
    return programs

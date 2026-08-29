"""Functional Omni-LUT AW/AA-GEMM lookup execution and K/V quantization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..quantization import (
    OmniQuantizedKV,
    ScaledCodebookTensor,
    VariableQuantizedTensor,
    quantize_oasis_weights,
    quantize_omni_kv,
)


Array = np.ndarray


@dataclass(frozen=True)
class OmniAWGemmResult:
    output: Array
    quantized_weights: ScaledCodebookTensor
    lut_entries_generated: int
    lookup_count: int


def omni_aw_lut_gemm(
    activation: Array,
    weights: Array,
    weight_bits: int = 4,
) -> OmniAWGemmResult:
    """Activation-weight GEMM through LGU products and weight-index queries.

    The LGU table is ``activation[k] * weight_centroid[code]``.  Per-output
    weight scales are applied after the indexed reduction.  The implementation
    intentionally gathers indices from that table; it never substitutes a
    dequantized matrix multiplication for the LUT datapath.
    """

    activation_data = np.asarray(activation, dtype=np.float32)
    weight_data = np.asarray(weights, dtype=np.float32)
    if activation_data.ndim == 1:
        activation_data = activation_data.reshape(1, -1)
    if activation_data.ndim != 2 or weight_data.ndim != 2:
        raise ValueError("Omni AW-GEMM requires [M,K] and [K,N] matrices")
    if activation_data.shape[1] != weight_data.shape[0]:
        raise ValueError("Omni AW-GEMM reduction dimensions differ")
    if weight_bits != 4:
        raise ValueError("the open Omni AW path supports packed W4 weights")
    quantized = quantize_oasis_weights(weight_data, weight_bits)
    rows, reduction = activation_data.shape
    outputs = weight_data.shape[1]
    # One scalar product table is shared by every output channel; the channel
    # scale is applied after reduction.  This is also the table exposed by the
    # synthesizable LGU/LUT-PE proxy.
    lut = activation_data[:, :, None] * quantized.codebook.centroids.reshape(1, 1, -1)
    result = np.empty((rows, outputs), dtype=np.float32)
    row_indices = np.arange(reduction)
    for output in range(outputs):
        selected = lut[:, row_indices, quantized.indices[:, output]]
        result[:, output] = np.sum(selected, axis=1) * quantized.scales[output]
    return OmniAWGemmResult(
        output=result,
        quantized_weights=quantized,
        lut_entries_generated=int(lut.size),
        lookup_count=int(rows * reduction * outputs),
    )


def _flatten_prefix(values: Array) -> Tuple[Array, Tuple[int, ...]]:
    prefix = tuple(values.shape[:-2])
    return values.reshape((-1,) + values.shape[-2:]), prefix


def qk_lut_gemm(query: Array, key: VariableQuantizedTensor) -> Array:
    """QK^T using LGU-created products and key indices, with no dequant GEMM."""

    query_data = np.asarray(query, dtype=np.float32)
    if query_data.shape[:-2] != key.indices.shape[:-2]:
        raise ValueError("query/key batch and head prefixes differ")
    if query_data.shape[-1] != key.indices.shape[-1]:
        raise ValueError("query/key head dimensions differ")
    query_flat, prefix = _flatten_prefix(query_data)
    key_flat, _ = _flatten_prefix(key.indices)
    groups, queries, dimension = query_flat.shape
    tokens = key_flat.shape[1]
    output = np.zeros((groups, queries, tokens), dtype=np.float32)
    for group in range(groups):
        for channel in range(dimension):
            bits = int(key.bits[channel])
            qmin = -(1 << (bits - 1))
            levels = np.arange(qmin, qmin + (1 << bits), dtype=np.float32)
            centroids = levels * key.scales[channel]
            centroids += key.corrections[channel, : 1 << bits]
            # LGU materializes this small [query, code] table once per channel.
            lut = query_flat[group, :, channel : channel + 1] * centroids.reshape(1, -1)
            selected = key_flat[group, :, channel].astype(np.int64)
            output[group] += lut[:, selected]
    return output.reshape(prefix + (queries, tokens))


def pv_lut_gemm(probability: Array, value: VariableQuantizedTensor) -> Array:
    """PV using a per-token compensated LUT and value indices."""

    probability_data = np.asarray(probability, dtype=np.float32)
    if probability_data.shape[:-2] != value.indices.shape[:-2]:
        raise ValueError("probability/value batch and head prefixes differ")
    if probability_data.shape[-1] != value.indices.shape[-2]:
        raise ValueError("probability KV length differs from value")
    probability_flat, prefix = _flatten_prefix(probability_data)
    value_flat, _ = _flatten_prefix(value.indices)
    groups, queries, tokens = probability_flat.shape
    dimension = value_flat.shape[-1]
    output = np.zeros((groups, queries, dimension), dtype=np.float32)
    for group in range(groups):
        for token in range(tokens):
            bits = int(value.bits[token])
            qmin = -(1 << (bits - 1))
            levels = np.arange(qmin, qmin + (1 << bits), dtype=np.float32)
            centroids = levels * value.scales[token]
            centroids += value.corrections[token, : 1 << bits]
            lut = probability_flat[group, :, token : token + 1] * centroids.reshape(1, -1)
            selected = value_flat[group, token, :].astype(np.int64)
            output[group] += lut[:, selected]
    return output.reshape(prefix + (queries, dimension))


def stable_softmax(values: Array, axis: int = -1) -> Array:
    data = np.asarray(values, dtype=np.float32)
    shifted = data - np.max(data, axis=axis, keepdims=True)
    exponent = np.exp(shifted.astype(np.float64))
    return (exponent / np.sum(exponent, axis=axis, keepdims=True)).astype(np.float32)


@dataclass(frozen=True)
class OmniAttentionResult:
    output: Array
    scores: Array
    probability: Array
    quantized_kv: OmniQuantizedKV
    key_cache_bytes: int
    value_cache_bytes: int


def packed_cache_bytes(tensor: VariableQuantizedTensor) -> int:
    axis = tensor.allocation_axis
    moved_shape = list(tensor.original_shape)
    slice_count = moved_shape.pop(axis)
    values_per_slice = int(np.prod(moved_shape))
    payload_bits = sum(int(bits) * values_per_slice for bits in tensor.bits)
    # FP16 scale and up to sixteen FP16 correction values per allocated slice.
    metadata_bytes = slice_count * (2 + 16 * 2)
    return int(math.ceil(payload_bits / 8.0) + metadata_bytes)


def omni_attention(
    query: Array,
    key: Array,
    value: Array,
    average_key_bits: float = 3.0,
    average_value_bits: float = 3.0,
    use_compensation: bool = True,
) -> OmniAttentionResult:
    query_data = np.asarray(query, dtype=np.float32)
    key_data = np.asarray(key, dtype=np.float32)
    value_data = np.asarray(value, dtype=np.float32)
    if query_data.shape[:-2] != key_data.shape[:-2] or key_data.shape != value_data.shape:
        raise ValueError("Omni attention prefixes or K/V shapes differ")
    quantized = quantize_omni_kv(
        key_data,
        value_data,
        average_key_bits,
        average_value_bits,
        use_compensation,
    )
    scores = qk_lut_gemm(query_data, quantized.key) / math.sqrt(query_data.shape[-1])
    probability = stable_softmax(scores)
    output = pv_lut_gemm(probability, quantized.value)
    return OmniAttentionResult(
        output,
        scores,
        probability,
        quantized,
        packed_cache_bytes(quantized.key),
        packed_cache_bytes(quantized.value),
    )

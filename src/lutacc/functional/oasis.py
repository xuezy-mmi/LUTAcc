"""Functional OASIS Cartesian-product LUT GEMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..quantization import (
    Codebook1D,
    OasisActivationTensor,
    ScaledCodebookTensor,
    quantize_oasis_activations,
    quantize_oasis_weights,
)


Array = np.ndarray


@dataclass(frozen=True)
class OasisGemmResult:
    output: Array
    lookahead_output: Array
    compensation: Array
    activation: OasisActivationTensor
    weight: ScaledCodebookTensor
    histogram_updates: int
    lut_entries: int


def cartesian_product_lut(
    activation_codebook: Codebook1D, weight_codebook: Codebook1D
) -> Array:
    return np.multiply.outer(
        activation_codebook.centroids, weight_codebook.centroids
    ).astype(np.float32)


def oasis_lut_gemm(
    activations: Array,
    weights: Array,
    activation_bits: int = 4,
    outlier_fraction: float = 0.01,
    activation_codebook: Optional[Codebook1D] = None,
    weight_codebook: Optional[Codebook1D] = None,
) -> OasisGemmResult:
    """Execute MxK times KxN entirely through index histograms and a LUT.

    The main branch counts concatenated activation/weight centroid indices.  The
    outlier branch multiplies only the activation residual at selected channels
    by dequantized W4 weights.  Their sum is mathematically identical to the
    conventional dynamic-outlier reference for the same quantized weights.
    """

    activation_matrix = np.asarray(activations, dtype=np.float32)
    if activation_matrix.ndim == 1:
        activation_matrix = activation_matrix.reshape(1, -1)
    weight_matrix = np.asarray(weights, dtype=np.float32)
    if activation_matrix.ndim != 2 or weight_matrix.ndim != 2:
        raise ValueError("OASIS GEMM expects rank-2 matrices")
    if activation_matrix.shape[1] != weight_matrix.shape[0]:
        raise ValueError("OASIS GEMM reduction dimensions do not match")
    weight_quantized = quantize_oasis_weights(weight_matrix, 4, weight_codebook)
    activation_quantized = quantize_oasis_activations(
        activation_matrix,
        activation_bits,
        activation_codebook,
        outlier_fraction,
    )
    base_lut = cartesian_product_lut(
        activation_quantized.quantized.codebook, weight_quantized.codebook
    )
    activation_levels, weight_levels = base_lut.shape
    flattened_lut = base_lut.reshape(-1)
    m, k = activation_matrix.shape
    n = weight_matrix.shape[1]
    lookahead = np.empty((m, n), dtype=np.float32)
    histogram_updates = 0
    for row in range(m):
        activation_indices = activation_quantized.quantized.indices[row].astype(np.int32)
        for column in range(n):
            weight_indices = weight_quantized.indices[:, column].astype(np.int32)
            combined = activation_indices * weight_levels + weight_indices
            counts = np.bincount(combined, minlength=activation_levels * weight_levels)
            base_sum = np.dot(counts.astype(np.float32), flattened_lut)
            lookahead[row, column] = (
                base_sum
                * activation_quantized.quantized.scales[row]
                * weight_quantized.scales[column]
            )
            histogram_updates += k
    dequantized_weight = weight_quantized.dequantize()
    compensation = activation_quantized.residuals @ dequantized_weight
    output = lookahead + compensation
    return OasisGemmResult(
        output.astype(np.float32),
        lookahead,
        compensation.astype(np.float32),
        activation_quantized,
        weight_quantized,
        histogram_updates,
        int(flattened_lut.size),
    )


def oasis_dynamic_outlier_reference(result: OasisGemmResult) -> Array:
    """Conventional separated-inlier GEMM for cross-checking look-ahead math."""

    quantized_activation = result.activation.quantized.dequantize()
    corrected_activation = quantized_activation + result.activation.residuals
    return corrected_activation @ result.weight.dequantize()

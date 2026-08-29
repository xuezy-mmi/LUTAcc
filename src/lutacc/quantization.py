"""Deterministic quantizers used by the OASIS and Omni-LUT reconstructions."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Codebook1D:
    centroids: Array

    def __post_init__(self) -> None:
        values = np.asarray(self.centroids, dtype=np.float32)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("a codebook requires at least two 1-D centroids")
        if not np.all(np.isfinite(values)):
            raise ValueError("centroids must be finite")
        if np.any(values[1:] < values[:-1]):
            raise ValueError("centroids must be sorted")
        object.__setattr__(self, "centroids", values)

    @property
    def levels(self) -> int:
        return int(self.centroids.size)

    @property
    def bits(self) -> int:
        bits = int(round(math.log(self.levels, 2)))
        if 1 << bits != self.levels:
            raise ValueError("codebook level count must be a power of two")
        return bits

    @property
    def boundaries(self) -> Array:
        return (self.centroids[:-1] + self.centroids[1:]) * np.float32(0.5)

    def encode(self, values: Array) -> Array:
        indices = np.searchsorted(self.boundaries, np.asarray(values), side="right")
        return indices.astype(np.uint8)

    def decode(self, indices: Array) -> Array:
        idx = np.asarray(indices)
        if idx.size and int(idx.max()) >= self.levels:
            raise ValueError("codebook index out of range")
        return self.centroids[idx]


def fit_kmeans_1d(
    values: Array,
    levels: int,
    sample_weights: Optional[Array] = None,
    iterations: int = 64,
) -> Codebook1D:
    """Fit sorted scalar K-Means using deterministic weighted quantiles.

    OASIS uses learned scalar codebooks.  A deterministic implementation is
    important because the same offline codebook feeds the compiler, functional
    model, and RTL vectors.
    """

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("K-Means input must contain finite values")
    if levels < 2 or levels & (levels - 1):
        raise ValueError("levels must be a power of two >= 2")
    if sample_weights is None:
        weights = np.ones_like(data)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape != data.shape or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("sample weights must be non-negative and match values")

    order = np.argsort(data, kind="stable")
    ordered = data[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = cumulative[-1]
    targets = (np.arange(levels, dtype=np.float64) + 0.5) * total / levels
    positions = np.searchsorted(cumulative, targets, side="left")
    positions = np.clip(positions, 0, ordered.size - 1)
    centroids = ordered[positions].copy()

    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        assignment = np.searchsorted(boundaries, data, side="right")
        updated = centroids.copy()
        for index in range(levels):
            mask = assignment == index
            if np.any(mask):
                local_weights = weights[mask]
                updated[index] = np.sum(data[mask] * local_weights) / np.sum(local_weights)
        updated.sort()
        if np.allclose(updated, centroids, rtol=0.0, atol=1e-7):
            centroids = updated
            break
        centroids = updated
    return Codebook1D(centroids.astype(np.float32))


@dataclass(frozen=True)
class ScaledCodebookTensor:
    indices: Array
    codebook: Codebook1D
    scales: Array
    scale_axis: int

    def dequantize(self) -> Array:
        decoded = self.codebook.decode(self.indices).astype(np.float32)
        shape = [1] * decoded.ndim
        axis = self.scale_axis % decoded.ndim
        shape[axis] = decoded.shape[axis]
        return decoded * np.asarray(self.scales, dtype=np.float32).reshape(shape)


def _safe_scales(values: Array, axis: int) -> Array:
    reduce_axes = tuple(index for index in range(values.ndim) if index != axis % values.ndim)
    scales = np.max(np.abs(values), axis=reduce_axes)
    return np.maximum(scales, np.finfo(np.float32).tiny).astype(np.float32)


def quantize_oasis_weights(
    weights: Array,
    bits: int = 4,
    codebook: Optional[Codebook1D] = None,
    force_zero_centroid: bool = False,
) -> ScaledCodebookTensor:
    matrix = np.asarray(weights, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("OASIS weights must have shape [K, N]")
    scales = _safe_scales(matrix, axis=1)
    normalized = matrix / scales.reshape(1, -1)
    learned = codebook or fit_kmeans_1d(normalized, 1 << bits)
    if codebook is None and force_zero_centroid:
        centroids = learned.centroids.copy()
        centroids[int(np.argmin(np.abs(centroids)))] = 0.0
        learned = Codebook1D(centroids)
    if learned.levels != 1 << bits:
        raise ValueError("weight codebook does not match requested precision")
    return ScaledCodebookTensor(learned.encode(normalized), learned, scales, 1)


@dataclass(frozen=True)
class OutlierSelection:
    mask: Array
    largest_indices: Tuple[Array, ...]
    smallest_indices: Tuple[Array, ...]
    comparisons: int
    k_each: int


def orizuru_select(values: Array, outlier_fraction: float) -> OutlierSelection:
    """Functional top/bottom-k selection with Orizuru's deterministic ties.

    The paper preserves the top 0.5% and bottom 0.5% for a 1% total outlier
    budget.  Equal FP16 values choose the lowest channel index (left child).
    """

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Orizuru input must be one vector or a matrix of tokens")
    if not 0.0 <= outlier_fraction < 1.0:
        raise ValueError("outlier_fraction must be in [0, 1)")
    length = matrix.shape[1]
    k_each = int(length * outlier_fraction * 0.5)
    mask = np.zeros(matrix.shape, dtype=bool)
    largest = []
    smallest = []
    channel_indices = np.arange(length)
    for row_index, row in enumerate(matrix):
        if k_each == 0:
            high = np.empty(0, dtype=np.int64)
            low = np.empty(0, dtype=np.int64)
        else:
            # lexsort uses the last key as primary. Index is the stable tie break.
            high_order = np.lexsort((channel_indices, -row))
            low_order = np.lexsort((channel_indices, row))
            high = high_order[:k_each].astype(np.int64)
            high_set = set(int(value) for value in high)
            low = np.asarray(
                [index for index in low_order if int(index) not in high_set][:k_each],
                dtype=np.int64,
            )
        mask[row_index, high] = True
        mask[row_index, low] = True
        largest.append(high)
        smallest.append(low)
    comparisons_per_token = int(1.5 * length + 2 * k_each * math.ceil(math.log(length, 2)))
    return OutlierSelection(
        mask=mask,
        largest_indices=tuple(largest),
        smallest_indices=tuple(smallest),
        comparisons=comparisons_per_token * matrix.shape[0],
        k_each=k_each,
    )


@dataclass(frozen=True)
class OasisActivationTensor:
    quantized: ScaledCodebookTensor
    residuals: Array
    outliers: OutlierSelection


def quantize_oasis_activations(
    activations: Array,
    bits: int = 4,
    codebook: Optional[Codebook1D] = None,
    outlier_fraction: float = 0.01,
    sample_weights: Optional[Array] = None,
) -> OasisActivationTensor:
    matrix = np.asarray(activations, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("OASIS activations must have shape [M, K]")
    selection = orizuru_select(matrix, outlier_fraction)
    inlier_values = np.where(selection.mask, 0.0, np.abs(matrix))
    scales = np.max(inlier_values, axis=1)
    fallback = np.max(np.abs(matrix), axis=1)
    scales = np.where(scales > 0, scales, fallback)
    scales = np.maximum(scales, np.finfo(np.float32).tiny).astype(np.float32)
    normalized = matrix / scales.reshape(-1, 1)
    if codebook is None:
        fit_values = normalized[~selection.mask]
        if fit_values.size == 0:
            fit_values = normalized.reshape(-1)
        learned = fit_kmeans_1d(fit_values, 1 << bits, sample_weights)
    else:
        learned = codebook
    if learned.levels != 1 << bits:
        raise ValueError("activation codebook does not match requested precision")
    indices = learned.encode(normalized)
    quantized = ScaledCodebookTensor(indices, learned, scales, 0)
    lookahead = quantized.dequantize()
    residuals = np.where(selection.mask, matrix - lookahead, 0.0).astype(np.float32)
    return OasisActivationTensor(quantized, residuals, selection)


@dataclass(frozen=True)
class ASBitAllocation:
    bits: Array
    requested_average_bits: float

    @property
    def actual_average_bits(self) -> float:
        return float(np.mean(self.bits))


def allocate_as_bits(
    sensitivity: Array,
    average_bits: float,
    minimum_bits: int = 2,
    maximum_bits: int = 4,
) -> ASBitAllocation:
    """Allocate an exact integer bit budget by marginal quantization benefit."""

    scores = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
    if scores.size == 0 or np.any(scores < 0) or not np.all(np.isfinite(scores)):
        raise ValueError("sensitivity must contain finite non-negative values")
    if not minimum_bits <= average_bits <= maximum_bits:
        raise ValueError("average bit budget is outside supported bounds")
    bits = np.full(scores.size, minimum_bits, dtype=np.uint8)
    target = int(round(average_bits * scores.size))
    extra = target - minimum_bits * scores.size
    heap = []

    def gain(index: int, current_bits: int) -> float:
        before = scores[index] * (2.0 ** (-2 * current_bits))
        after = scores[index] * (2.0 ** (-2 * (current_bits + 1)))
        return before - after

    for index in range(scores.size):
        heapq.heappush(heap, (-gain(index, minimum_bits), index, minimum_bits))
    for _ in range(extra):
        while heap:
            _, index, recorded = heapq.heappop(heap)
            if int(bits[index]) == recorded and recorded < maximum_bits:
                break
        else:
            raise RuntimeError("AS-Bit budget exceeds representable allocation")
        bits[index] += 1
        updated = int(bits[index])
        if updated < maximum_bits:
            heapq.heappush(heap, (-gain(index, updated), index, updated))
    return ASBitAllocation(bits, float(average_bits))


@dataclass(frozen=True)
class VariableQuantizedTensor:
    indices: Array
    bits: Array
    scales: Array
    corrections: Array
    allocation_axis: int
    original_shape: Tuple[int, ...]

    @property
    def maximum_levels(self) -> int:
        return int(self.corrections.shape[1])

    def signed_levels(self, slice_index: int) -> Array:
        bits = int(self.bits[slice_index])
        qmin = -(1 << (bits - 1))
        count = 1 << bits
        return np.arange(qmin, qmin + count, dtype=np.float32)

    def dequantize(self, compensated: bool = True) -> Array:
        moved = np.moveaxis(self.indices, self.allocation_axis, -1)
        result = np.empty(moved.shape, dtype=np.float32)
        for slice_index in range(moved.shape[-1]):
            bits = int(self.bits[slice_index])
            qmin = -(1 << (bits - 1))
            local_indices = moved[..., slice_index].astype(np.int32)
            values = (local_indices + qmin).astype(np.float32) * self.scales[slice_index]
            if compensated:
                values = values + self.corrections[slice_index, local_indices]
            result[..., slice_index] = values
        return np.moveaxis(result, -1, self.allocation_axis)


def _slice_sensitivity(values: Array, axis: int) -> Array:
    moved = np.moveaxis(values, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    # Second moment weights high-energy channels/tokens without a model-specific
    # Hessian. A caller may supply calibrated Fisher scores instead.
    return np.mean(np.square(flat.astype(np.float64)), axis=0) + 1e-12


def quantize_variable_symmetric(
    values: Array,
    average_bits: float,
    allocation_axis: int,
    sensitivity: Optional[Array] = None,
    use_compensation: bool = True,
) -> VariableQuantizedTensor:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim < 2:
        raise ValueError("variable quantization requires at least two dimensions")
    axis = allocation_axis % data.ndim
    moved = np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    scores = _slice_sensitivity(data, axis) if sensitivity is None else sensitivity
    allocation = allocate_as_bits(scores, average_bits)
    indices = np.empty(flat.shape, dtype=np.uint8)
    scales = np.empty(flat.shape[1], dtype=np.float32)
    corrections = np.zeros((flat.shape[1], 1 << 4), dtype=np.float32)
    for slice_index in range(flat.shape[1]):
        bits = int(allocation.bits[slice_index])
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        max_abs = float(np.max(np.abs(flat[:, slice_index])))
        scale = max(max_abs / max(abs(qmin), abs(qmax)), np.finfo(np.float32).tiny)
        signed = np.clip(np.rint(flat[:, slice_index] / scale), qmin, qmax).astype(np.int32)
        local_indices = (signed - qmin).astype(np.uint8)
        indices[:, slice_index] = local_indices
        scales[slice_index] = scale
        if use_compensation:
            reconstruction = signed.astype(np.float32) * scale
            residual = flat[:, slice_index] - reconstruction
            for code in range(1 << bits):
                mask = local_indices == code
                if np.any(mask):
                    corrections[slice_index, code] = float(np.mean(residual[mask]))
    shaped = indices.reshape(moved.shape)
    shaped = np.moveaxis(shaped, -1, axis)
    return VariableQuantizedTensor(
        shaped,
        allocation.bits,
        scales,
        corrections,
        axis,
        tuple(data.shape),
    )


@dataclass(frozen=True)
class OmniQuantizedKV:
    key: VariableQuantizedTensor
    value: VariableQuantizedTensor


def quantize_omni_kv(
    key: Array,
    value: Array,
    average_key_bits: float = 3.0,
    average_value_bits: float = 3.0,
    use_compensation: bool = True,
) -> OmniQuantizedKV:
    key_data = np.asarray(key, dtype=np.float32)
    value_data = np.asarray(value, dtype=np.float32)
    if key_data.shape != value_data.shape or key_data.ndim < 2:
        raise ValueError("key and value tensors must have equal [..., tokens, dim] shape")
    # The open reconstruction uses the hardware-friendly directions described in
    # the project brief: K is allocated per channel; V is allocated per token.
    key_quantized = quantize_variable_symmetric(
        key_data,
        average_key_bits,
        allocation_axis=key_data.ndim - 1,
        use_compensation=use_compensation,
    )
    value_quantized = quantize_variable_symmetric(
        value_data,
        average_value_bits,
        allocation_axis=value_data.ndim - 2,
        use_compensation=use_compensation,
    )
    return OmniQuantizedKV(key_quantized, value_quantized)

import numpy as np

from lutacc.quantization import (
    allocate_as_bits,
    fit_kmeans_1d,
    orizuru_select,
    quantize_variable_symmetric,
    quantize_oasis_weights,
)


def test_scalar_kmeans_is_sorted_and_deterministic():
    rng = np.random.default_rng(4)
    values = rng.normal(size=1000).astype(np.float32)
    first = fit_kmeans_1d(values, 16)
    second = fit_kmeans_1d(values, 16)
    assert np.all(first.centroids[:-1] <= first.centroids[1:])
    np.testing.assert_array_equal(first.centroids, second.centroids)


def test_orizuru_selects_both_tails_and_breaks_ties_left():
    values = np.arange(200, dtype=np.float32)
    values[0] = -100.0
    values[1] = -100.0
    values[-1] = 1000.0
    values[-2] = 1000.0
    selection = orizuru_select(values, 0.02)
    # floor(200 * 2% / 2) selects two values per tail.
    assert selection.k_each == 2
    assert list(selection.smallest_indices[0]) == [0, 1]
    assert list(selection.largest_indices[0]) == [198, 199]
    assert selection.mask.sum() == 4


def test_as_bit_allocator_honors_exact_budget_and_sensitivity():
    sensitivity = np.asarray([100.0, 10.0, 1.0, 0.1])
    allocation = allocate_as_bits(sensitivity, 3.0)
    assert allocation.bits.sum() == 12
    assert allocation.actual_average_bits == 3.0
    assert allocation.bits[0] >= allocation.bits[-1]


def test_quantization_compensation_cannot_increase_training_mse():
    rng = np.random.default_rng(18)
    values = rng.normal(loc=0.3, scale=1.7, size=(9, 23)).astype(np.float32)
    quantized = quantize_variable_symmetric(values, 2.5, allocation_axis=1)
    raw = quantized.dequantize(compensated=False)
    compensated = quantized.dequantize(compensated=True)
    raw_mse = np.mean(np.square(values - raw))
    compensated_mse = np.mean(np.square(values - compensated))
    assert compensated_mse <= raw_mse + 1e-8


def test_oasis_weight_codebook_contains_exact_zero():
    rng = np.random.default_rng(31)
    quantized = quantize_oasis_weights(
        rng.normal(size=(128, 32)).astype(np.float32), force_zero_centroid=True
    )
    assert np.count_nonzero(quantized.codebook.centroids == 0.0) == 1

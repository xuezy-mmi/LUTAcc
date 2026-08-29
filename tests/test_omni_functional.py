import math

import numpy as np

from lutacc.functional.omni import (
    omni_attention,
    omni_aw_lut_gemm,
    packed_cache_bytes,
    pv_lut_gemm,
    qk_lut_gemm,
    stable_softmax,
)
from lutacc.quantization import quantize_omni_kv


def test_aw_lut_path_equals_quantized_weight_reference():
    rng = np.random.default_rng(41)
    activation = rng.normal(size=(3, 17)).astype(np.float32)
    weight = rng.normal(size=(17, 11)).astype(np.float32)
    result = omni_aw_lut_gemm(activation, weight)
    reference = activation @ result.quantized_weights.dequantize()
    np.testing.assert_allclose(result.output, reference, rtol=2e-5, atol=2e-5)
    assert result.lut_entries_generated == 3 * 17 * 16
    assert result.lookup_count == 3 * 17 * 11


def test_qk_and_pv_lut_paths_equal_compensated_dequant_reference():
    rng = np.random.default_rng(9)
    query = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
    key = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    value = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    quantized = quantize_omni_kv(key, value)

    scores = qk_lut_gemm(query, quantized.key)
    reference_scores = query @ np.swapaxes(quantized.key.dequantize(), -1, -2)
    np.testing.assert_allclose(scores, reference_scores, rtol=2e-5, atol=2e-5)

    probability = stable_softmax(scores / math.sqrt(query.shape[-1]))
    output = pv_lut_gemm(probability, quantized.value)
    reference_output = probability @ quantized.value.dequantize()
    np.testing.assert_allclose(output, reference_output, rtol=2e-5, atol=2e-5)


def test_omni_attention_is_finite_and_compresses_realistic_kv_cache():
    rng = np.random.default_rng(22)
    query = rng.normal(size=(1, 2, 1, 32)).astype(np.float32)
    key = rng.normal(size=(1, 2, 128, 32)).astype(np.float32)
    value = rng.normal(size=(1, 2, 128, 32)).astype(np.float32)
    result = omni_attention(query, key, value)
    assert np.isfinite(result.output).all()
    fp16_bytes = key.size * 2 + value.size * 2
    assert result.key_cache_bytes + result.value_cache_bytes < fp16_bytes


def test_quantization_compensation_improves_attention_output_on_fit_tensor():
    rng = np.random.default_rng(3)
    query = rng.normal(size=(1, 1, 4, 16)).astype(np.float32)
    key = rng.normal(loc=0.4, size=(1, 1, 12, 16)).astype(np.float32)
    value = rng.normal(loc=-0.2, size=(1, 1, 12, 16)).astype(np.float32)
    compensated = omni_attention(query, key, value, 2.0, 2.0, True)
    raw = omni_attention(query, key, value, 2.0, 2.0, False)
    full_scores = query @ np.swapaxes(key, -1, -2) / math.sqrt(16)
    full = stable_softmax(full_scores) @ value
    assert np.linalg.norm(compensated.output - full) <= np.linalg.norm(raw.output - full) + 1e-5

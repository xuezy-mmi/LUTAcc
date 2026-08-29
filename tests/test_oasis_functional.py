import numpy as np

from lutacc.functional.oasis import oasis_dynamic_outlier_reference, oasis_lut_gemm


def test_cartesian_lut_plus_compensation_matches_dynamic_reference():
    rng = np.random.default_rng(7)
    activation = rng.normal(size=(2, 64)).astype(np.float32)
    activation[0, 3] = 18.0
    activation[0, 9] = -21.0
    weight = rng.normal(size=(64, 13)).astype(np.float32)
    result = oasis_lut_gemm(activation, weight, outlier_fraction=0.0625)
    reference = oasis_dynamic_outlier_reference(result)
    np.testing.assert_allclose(result.output, reference, rtol=2e-5, atol=5e-5)
    assert result.lut_entries == 256
    assert result.histogram_updates == 2 * 64 * 13


def test_error_compensation_repairs_selected_activation_residuals():
    rng = np.random.default_rng(12)
    activation = rng.normal(size=(1, 256)).astype(np.float32)
    activation[0, 4] = 40.0
    activation[0, 31] = -35.0
    weight = rng.normal(size=(256, 17)).astype(np.float32)
    result = oasis_lut_gemm(activation, weight, outlier_fraction=0.02)
    target = activation @ result.weight.dequantize()
    lookahead_error = np.linalg.norm(result.lookahead_output - target)
    corrected_error = np.linalg.norm(result.output - target)
    assert corrected_error < lookahead_error


def test_w4a3_uses_128_entry_cartesian_lut():
    rng = np.random.default_rng(2)
    result = oasis_lut_gemm(
        rng.normal(size=(1, 32)).astype(np.float32),
        rng.normal(size=(32, 4)).astype(np.float32),
        activation_bits=3,
    )
    assert result.lut_entries == 128

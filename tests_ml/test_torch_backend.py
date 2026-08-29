import torch

from lutacc.torch_backend import (
    ActivationCalibration,
    OasisActivationHooks,
    fit_torch_kmeans_1d,
    quantize_linear_weight_oasis,
    uniform_sample_positions,
)


def test_small_fisher_weights_do_not_collapse_centroids():
    values = torch.linspace(-1, 1, 4096)
    weights = torch.linspace(1e-9, 1e-7, 4096)
    codebook = fit_torch_kmeans_1d(values, 16, sample_weights=weights)
    assert torch.unique(codebook.centroids).numel() == 16
    assert codebook.centroids[0] < -0.8
    assert codebook.centroids[-1] > 0.8


def test_large_tensor_sampling_never_emits_end_index():
    length = (1 << 24) + 123
    positions = uniform_sample_positions(length, 65536, torch.device("cpu"))
    assert positions[0] == 0
    assert positions[-1] == length - 1
    assert int(positions.max()) < length


def test_gradient_fisher_calibration_and_runtime_hook():
    network = torch.nn.Sequential(
        torch.nn.Linear(32, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 8),
    )
    modules = [("first", network[0]), ("second", network[2])]
    calibration = ActivationCalibration(
        modules,
        outlier_fraction=0.0625,
        maximum_values_per_module=1024,
        gradient_fisher=True,
    )
    activation = torch.randn(8, 32, requires_grad=True)
    network(activation).square().mean().backward()
    calibration.close()
    codebooks = calibration.fit(4)
    assert set(codebooks) == {"first", "second"}
    assert all(torch.unique(item.centroids).numel() == 16 for item in codebooks.values())

    hooks = OasisActivationHooks(modules, codebooks, outlier_fraction=0.0625)
    output = network(torch.randn(4, 32))
    hooks.close()
    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()


def test_runtime_hook_chunking_matches_single_chunk():
    torch.manual_seed(7)
    activation = torch.randn(3, 5, 32)
    codebook = fit_torch_kmeans_1d(torch.linspace(-1, 1, 4096), 16)

    chunked_linear = torch.nn.Linear(32, 8, bias=False)
    reference_linear = torch.nn.Linear(32, 8, bias=False)
    reference_linear.load_state_dict(chunked_linear.state_dict())
    chunked = OasisActivationHooks(
        [("linear", chunked_linear)],
        {"linear": codebook},
        outlier_fraction=0.125,
        maximum_working_set_elements=64,
    )
    unchunked = OasisActivationHooks(
        [("linear", reference_linear)],
        {"linear": codebook},
        outlier_fraction=0.125,
        maximum_working_set_elements=activation.numel(),
    )
    chunked_output = chunked_linear(activation)
    reference_output = reference_linear(activation)
    chunked.close()
    unchunked.close()

    torch.testing.assert_close(chunked_output, reference_output, rtol=0, atol=0)


def test_fixed_zero_lloyd_refit_preserves_zero_and_reduces_weight_rmse():
    torch.manual_seed(29)
    base = torch.nn.Linear(96, 64, bias=False)
    pinned = torch.nn.Linear(96, 64, bias=False)
    refitted = torch.nn.Linear(96, 64, bias=False)
    pinned.load_state_dict(base.state_dict())
    refitted.load_state_dict(base.state_dict())
    pinned_result = quantize_linear_weight_oasis(
        pinned, force_zero_centroid=True
    )
    refitted_result = quantize_linear_weight_oasis(
        refitted,
        force_zero_centroid=True,
        zero_centroid_refit_iterations=8,
    )
    assert 0.0 in refitted_result["centroids"]
    assert refitted_result["rmse"] <= pinned_result["rmse"]

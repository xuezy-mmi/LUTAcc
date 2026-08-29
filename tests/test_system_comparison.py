import pytest

from lutacc.schema import model_spec
from lutacc.sim.system import (
    compare_oasis_figlut_decode,
    compare_oasis_gpu_decode,
    compare_oasis_lookahead,
)


PAPER_MODELS = [
    model_spec("llama2-7b"),
    model_spec("llama2-13b"),
    model_spec("llama-30b"),
    model_spec("llama2-70b"),
]


@pytest.mark.parametrize(
    "activation_bits,target_speedup,target_efficiency",
    [(4, 3.00, 1.44), (3, 3.15, 1.51)],
)
def test_oasis_figlut_headlines_within_ten_percent(
    activation_bits, target_speedup, target_efficiency
):
    comparison = compare_oasis_figlut_decode(PAPER_MODELS, activation_bits)
    assert comparison["geomean_speedup"] == pytest.approx(target_speedup, rel=0.10)
    assert comparison["geomean_energy_efficiency_improvement"] == pytest.approx(
        target_efficiency, rel=0.10
    )


def test_comparison_is_pointwise_not_only_a_geometric_mean():
    comparison = compare_oasis_figlut_decode(PAPER_MODELS, 4)
    assert len(comparison["points"]) == 4
    for point in comparison["points"]:
        assert point["speedup"] > 1
        assert point["energy_efficiency_improvement"] > 1


def test_calibrated_gpu_rooflines_cover_all_reported_headlines():
    targets = {
        4: {"a100_fp16": (5.41, 198.10), "quarot_w4a4": (3.12, 108.80)},
        3: {"a100_fp16": (5.67, 206.53), "quarot_w4a4": (3.27, 113.56)},
    }
    for bits in (4, 3):
        result = compare_oasis_gpu_decode(PAPER_MODELS, bits)
        assert result["comparison_evidence"].endswith("not_native")
        for baseline, (speed, efficiency) in targets[bits].items():
            assert result["geomeans"][baseline]["speedup"] == pytest.approx(
                speed, rel=0.10
            )
            assert result["geomeans"][baseline][
                "energy_efficiency_improvement"
            ] == pytest.approx(efficiency, rel=0.10)


@pytest.mark.parametrize("bits,target", [(4, 1.16), (3, 1.18)])
def test_lookahead_ablation_matches_reported_throughput_gain(bits, target):
    result = compare_oasis_lookahead(model_spec("llama2-7b"), bits)
    assert result["throughput_improvement"] == pytest.approx(target, rel=0.10)
    assert result["lookahead"]["cycles_per_token"] < result[
        "conventional_dynamic_detection"
    ]["cycles_per_token"]

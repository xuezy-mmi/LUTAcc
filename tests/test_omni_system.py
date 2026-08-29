import pytest

from lutacc.schema import model_spec
from lutacc.sim.omni_system import (
    compare_omni_figlut_long_context,
    simulate_omni_transformer,
)


def test_omni_reported_energy_range_within_ten_percent():
    result = compare_omni_figlut_long_context(model_spec("llama2-7b"))
    assert result["minimum_improvement"] == pytest.approx(1.25, rel=0.10)
    assert result["maximum_improvement"] == pytest.approx(1.91, rel=0.10)
    values = [point["energy_efficiency_improvement"] for point in result["points"]]
    assert values == sorted(values)


def test_omni_kv_bytes_grow_linearly_and_remain_below_fp16():
    result = compare_omni_figlut_long_context(model_spec("llama2-7b"))
    points = result["points"]
    assert points[1]["omni_kv_bytes"] == 2 * points[0]["omni_kv_bytes"]
    assert all(point["omni_kv_bytes"] < point["figlut_kv_bytes"] for point in points)


def test_omni_system_includes_aw_and_aa_paths_and_scales_with_context():
    model = model_spec("llama2-7b")
    short = simulate_omni_transformer(model, "decode", 1, 128)
    long = simulate_omni_transformer(model, "decode", 1, 8192)
    assert [item["operation"] for item in short.layer_breakdown] == [
        "qkv",
        "attention_qk_softmax_pv",
        "o_proj",
        "gate_up",
        "down",
    ]
    assert sum("omni-lut-aw" == item["design"] for item in short.layer_breakdown) == 4
    assert long.cycles > short.cycles
    assert long.dram_bytes > short.dram_bytes

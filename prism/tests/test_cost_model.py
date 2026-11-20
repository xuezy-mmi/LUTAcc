import pytest

from costmodel import (
    ArrayConfig,
    GemmShape,
    LutConfig,
    Primitive,
    PrecisionSpec,
    PrismCostModel,
)


@pytest.fixture
def common():
    return dict(
        shape=GemmShape(32, 64, 128),
        weight=PrecisionSpec(4),
        activation=PrecisionSpec(8),
        accumulator=PrecisionSpec(32),
        array=ArrayConfig(8, 8, tile_k=16),
    )


def test_mac_and_lut_have_expected_metadata(common):
    model = PrismCostModel()
    mac = model.estimate(**common, primitive=Primitive.MAC)
    lut = model.estimate(**common, primitive=Primitive.LUT, lut=LutConfig(group_size=1))
    assert mac.mac_ops == 32 * 64 * 128
    assert mac.lut_lookups == 0
    assert lut.mac_ops == 0
    assert lut.lut_lookups > 0
    assert lut.lut_entries > 0


def test_lut_sharing_changes_storage_not_functional_work(common):
    model = PrismCostModel()
    per_pe = model.estimate(**common, primitive=Primitive.LUT, lut=LutConfig(sharing="per_pe"))
    global_ = model.estimate(**common, primitive=Primitive.LUT, lut=LutConfig(sharing="global"))
    assert per_pe.lut_lookups == global_.lut_lookups
    assert per_pe.lut_storage_bits > global_.lut_storage_bits
    assert per_pe.area_mm2 > global_.area_mm2


def test_array_scale_affects_latency_and_utilization(common):
    model = PrismCostModel()
    config = {key: value for key, value in common.items() if key != "array"}
    small = model.estimate(**config, array=ArrayConfig(4, 4), primitive=Primitive.MAC)
    large = model.estimate(**config, array=ArrayConfig(16, 16), primitive=Primitive.MAC)
    assert large.latency_cycles <= small.latency_cycles
    assert large.breakdown["array_pes"] > small.breakdown["array_pes"]


def test_w4a16_requires_digit_decomposition(common):
    model = PrismCostModel()
    estimate = model.estimate(
        **{**common, "activation": PrecisionSpec(16)},
        primitive=Primitive.LUT,
        lut=LutConfig(group_size=1, digit_bits=4),
    )
    assert estimate.breakdown["lut_entries_per_table"] == 16
    assert estimate.lut_lookups > 0


def test_invalid_lut_index_is_rejected(common):
    with pytest.raises(ValueError, match="LUT index"):
        PrismCostModel().estimate(
            **common,
            primitive=Primitive.LUT,
            lut=LutConfig(group_size=8, digit_bits=4, max_index_bits=8),
        )

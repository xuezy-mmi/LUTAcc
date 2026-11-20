import pytest

from costmodel import ArrayConfig, GemmShape, PrecisionSpec, Primitive, PrismCostModel
from dse import (
    DseRequest,
    NoFeasibleDesignError,
    PrismDse,
    ResourceConstraints,
    SearchSpace,
)


@pytest.fixture
def compact_space():
    return SearchSpace(
        array_rows=(4, 8),
        array_cols=(4, 8),
        tile_ks=(8, 16),
        dataflows=("output_stationary", "weight_stationary"),
        lut_group_sizes=(1,),
        lut_digit_bits=(4,),
        lut_sharing=("global",),
        hybrid_lut_fractions=(0.5,),
    )


@pytest.fixture
def base_request(compact_space):
    return DseRequest(
        shape=GemmShape(8, 8, 16),
        weight_precision=PrecisionSpec(4),
        input_precision=PrecisionSpec(8),
        output_precision=PrecisionSpec(8),
        search_space=compact_space,
    )


def test_dse_returns_generator_parameters_and_cost_report(base_request):
    result = PrismDse().search(base_request)
    payload = result.as_dict()
    assert result.selected.array.dataflow in {"output_stationary", "weight_stationary"}
    assert result.report.primitive == result.selected.primitive
    assert result.report.area_mm2 > 0
    assert result.feasible_candidates > 0
    assert payload["selected_accelerator"]["array"]["dataflow"] == result.selected.array.dataflow
    assert payload["cost_report"]["latency_cycles"] == result.report.latency_cycles


def test_dse_respects_hard_resource_limits(base_request):
    constrained = DseRequest(
        shape=base_request.shape,
        weight_precision=base_request.weight_precision,
        input_precision=base_request.input_precision,
        output_precision=base_request.output_precision,
        constraints=ResourceConstraints(max_area_mm2=0.012, max_lut_storage_bits=1000),
        search_space=base_request.search_space,
    )
    result = PrismDse().search(constrained)
    assert result.report.area_mm2 <= 0.012
    assert result.report.lut_storage_bits <= 1000
    assert result.selected.primitive.value == "mac"


def test_dse_reports_infeasible_constraint_set(base_request):
    impossible = DseRequest(
        shape=base_request.shape,
        weight_precision=base_request.weight_precision,
        input_precision=base_request.input_precision,
        output_precision=base_request.output_precision,
        constraints=ResourceConstraints(max_area_mm2=1.0e-9),
        search_space=base_request.search_space,
    )
    with pytest.raises(NoFeasibleDesignError, match="no design satisfies"):
        PrismDse().search(impossible)


def test_derived_accumulator_covers_reduction_growth(request):
    request = DseRequest(
        shape=GemmShape(4, 4, 256),
        weight_precision=PrecisionSpec(4),
        input_precision=PrecisionSpec(8),
        output_precision=PrecisionSpec(8),
    )
    assert request.resolved_accumulator_precision().bits == 20


def test_dataflows_produce_different_movement_costs():
    model = PrismCostModel()
    common = dict(
        shape=GemmShape(32, 8, 16),
        weight=PrecisionSpec(4),
        activation=PrecisionSpec(8),
        accumulator=PrecisionSpec(16),
        primitive=Primitive.MAC,
    )
    costs = {
        model.estimate(**common, array=ArrayConfig(4, 4, dataflow=dataflow)).breakdown[
            "movement_bits"
        ]
        for dataflow in ("output_stationary", "weight_stationary", "input_stationary")
    }
    assert len(costs) > 1

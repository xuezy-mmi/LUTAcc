import pytest

from lutacc.schema import AttentionShape, GemmShape
from lutacc.sim.event import EventScheduler, EventSpec, makespan
from lutacc.sim.oasis import OasisCycleModel, simulate_oasis_gemm
from lutacc.sim.omni import OmniCycleModel, simulate_omni_attention


def test_oasis_paper_pipeline_anchor_is_exact():
    result = simulate_oasis_gemm(GemmShape(1, 4096, 4096), 4, 0.01)
    ledger = result.metadata["cycle_ledger"]
    assert ledger["activation_quantization"] == 2048
    assert ledger["concat"] == 256
    assert ledger["index_distribution"] == 1024
    assert ledger["reduction"] == 2304
    assert ledger["outlier_detection"] == 1344
    assert ledger["error_compensation"] == 1536
    assert ledger["merge"] == 256
    assert result.cycles == 2560
    assert result.latency_us == pytest.approx(5.12)


def test_oasis_anchor_memory_breakdown_matches_reported_plot():
    result = simulate_oasis_gemm(GemmShape(1, 4096, 4096), 4, 0.01)
    measured = result.metadata["sram_traffic_percent"]
    targets = {
        "weight_index_buffer": 76.0,
        "lut": 19.2,
        "activation_index_buffer": 4.8,
        "output_buffer": 0.1,
    }
    for name, target in targets.items():
        assert measured[name] == pytest.approx(target, abs=0.25)


def test_oasis_anchor_stage_energy_breakdown_is_auditable():
    result = simulate_oasis_gemm(GemmShape(1, 4096, 4096), 4, 0.01)
    measured = result.metadata["stage_energy_percent"]
    targets = {
        "activation_quantization": 14.7,
        "index_distribution": 7.4,
        "error_compensation": 3.6,
        "concat": 1.8,
        "reduction": 33.1,
        "outlier_detection": 17.4,
        "merge": 22.1,
    }
    assert sum(measured.values()) == pytest.approx(100.0)
    for name, target in targets.items():
        assert measured[name] == pytest.approx(target, abs=0.25)


def test_oasis_precision_and_outlier_trends_follow_mechanism():
    shape = GemmShape(1, 4096, 4096)
    a4 = simulate_oasis_gemm(shape, 4, 0.01)
    a3 = simulate_oasis_gemm(shape, 3, 0.01)
    heavy = simulate_oasis_gemm(shape, 4, 0.10)
    assert a3.cycles < a4.cycles
    assert heavy.cycles > a4.cycles
    assert heavy.metadata["cycle_ledger"]["outlier_branch"] > heavy.metadata["cycle_ledger"]["main_branch"]


def test_event_scheduler_enforces_resource_serialization_and_dependency():
    events = [
        EventSpec("a", "pe", 5),
        EventSpec("b", "pe", 7),
        EventSpec("c", "merge", 3, ("a", "b")),
    ]
    stages = EventScheduler().schedule(events)
    assert stages[0].start_cycle == 0
    assert stages[1].start_cycle == 5
    assert stages[2].start_cycle == 12
    assert makespan(stages) == 15


def test_omni_array_and_phase_metadata_are_explicit():
    model = OmniCycleModel.from_file()
    assert (model.array_rows, model.array_cols, model.lut_pes) == (32, 4, 128)
    prefill = simulate_omni_attention(AttentionShape(1, 8, 64, 64, 128), "prefill")
    decode = simulate_omni_attention(AttentionShape(1, 8, 1, 2048, 128), "decode")
    assert prefill.metadata["dataflow"] == "prefill_ws"
    assert decode.metadata["dataflow"] == "decode_os_v"
    assert prefill.cycles > 0 and decode.cycles > 0
    assert decode.metadata["configuration_evidence"] == "inferred"

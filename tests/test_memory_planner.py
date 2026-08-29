from lutacc.compiler import (
    compile_omni_attention,
    compile_oasis_gemm,
    plan_omni_kv_tiles,
    plan_sram,
)
from lutacc.schema import AttentionShape, GemmShape


def test_oasis_anchor_sram_lifetimes_fit_reported_buffers():
    program = compile_oasis_gemm(GemmShape(1, 4096, 4096))
    plan = plan_sram(program, 64 * 1024 + 16 * 1024)
    assert plan.spill_bytes == 0
    assert plan.peak_bytes <= 80 * 1024
    allocations = {item.tensor: item for item in plan.allocations}
    assert allocations["lookahead_output"].offset != allocations["compensation"].offset


def test_lifetime_planner_reuses_non_overlapping_storage():
    program = compile_oasis_gemm(GemmShape(1, 128, 64))
    plan = plan_sram(program, 4096)
    assert plan.peak_bytes < sum(item.size_bytes for item in plan.allocations)


def test_long_context_omni_kv_generates_multiple_tiles():
    shape = AttentionShape(1, 32, 1, 16384, 128)
    plan = plan_omni_kv_tiles(shape, 3.0, 3.0, 2 * 1024 * 1024)
    assert plan["requires_dram_tiling"] is True
    assert plan["tiles"] > 1
    assert 0 < plan["last_tile_tokens"] <= plan["tile_tokens"]


def test_short_context_omni_kv_is_resident():
    shape = AttentionShape(1, 8, 1, 128, 128)
    plan = plan_omni_kv_tiles(shape, 3.0, 3.0, 2 * 1024 * 1024)
    assert plan["tiles"] == 1
    assert plan["requires_dram_tiling"] is False

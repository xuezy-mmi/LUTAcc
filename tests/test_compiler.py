import json

import pytest

from lutacc.compiler import compile_omni_attention, compile_oasis_gemm
from lutacc.schema import AttentionShape, GemmShape


def test_oasis_ir_contains_both_parallel_branches(tmp_path):
    program = compile_oasis_gemm(GemmShape(1, 4096, 4096))
    names = {operation.name for operation in program.operations}
    assert {"lut_reduce", "detect_outliers", "error_compensate", "merge_branches"} <= names
    merge = next(operation for operation in program.operations if operation.name == "merge_branches")
    assert set(merge.dependencies) == {"lut_reduce", "error_compensate"}
    target = tmp_path / "program.json"
    program.write_json(str(target))
    document = json.loads(target.read_text())
    assert document["metadata"]["branch_overlap"] is True
    assert document["design"] == "oasis"


def test_omni_ir_selects_phase_adaptive_dataflow():
    shape = AttentionShape(1, 8, 64, 64, 128)
    prefill = compile_omni_attention(shape, "prefill")
    decode = compile_omni_attention(AttentionShape(1, 8, 1, 2048, 128), "decode")
    assert prefill.metadata["phase_adaptive_dataflow"] == "prefill_ws"
    assert decode.metadata["phase_adaptive_dataflow"] == "decode_os_v"
    assert any(operation.kind == "activation_activation_lut_gemm" for operation in decode.operations)
    assert any(operation.kind == "as_bit_allocate" for operation in decode.operations)


def test_compiler_rejects_unsupported_precision():
    with pytest.raises(ValueError, match="W4A3 and W4A4"):
        compile_oasis_gemm(GemmShape(1, 16, 16), weight_bits=3)

import json

import pytest

from lutacc.cli import main
from lutacc.compiler import compile_omni_attention, compile_omni_gemm, compile_oasis_gemm
from lutacc.compiler.ir import Program
from lutacc.schema import AttentionShape, GemmShape
from lutacc.sim import simulate_program


def test_oasis_ir_round_trip_drives_exact_anchor(tmp_path):
    path = tmp_path / "oasis.json"
    compile_oasis_gemm(GemmShape(1, 4096, 4096)).write_json(str(path))
    restored = Program.read_json(str(path))
    result = simulate_program(restored)
    assert result.cycles == 2560
    assert result.metadata["compiler_simulator_contract"] == "validated"


def test_omni_ir_round_trip_preserves_decode_dataflow(tmp_path):
    path = tmp_path / "omni.json"
    trace = tmp_path / "trace.json"
    compile_omni_attention(
        AttentionShape(1, 8, 1, 2048, 128), "decode"
    ).write_json(str(path))
    assert main(["simulate-ir", "--program", str(path), "--trace", str(trace)]) == 0
    document = json.loads(trace.read_text())
    assert document["metadata"]["dataflow"] == "decode_os_v"
    assert document["metadata"]["compiler_simulator_contract"] == "validated"


def test_omni_aw_ir_drives_weight_stationary_cycle_path(tmp_path):
    path = tmp_path / "omni_aw.json"
    compile_omni_gemm(GemmShape(2, 128, 96)).write_json(str(path))
    result = simulate_program(Program.read_json(str(path)), resident_weights=False)
    assert result.design == "omni-lut-aw"
    assert result.metadata["dataflow"] == "weight_stationary"
    assert result.metadata["lut_lookups"] == 2 * 128 * 96


def test_ir_simulator_rejects_mechanism_tampering():
    program = compile_oasis_gemm(GemmShape(1, 64, 64))
    program.operations[0] = type(program.operations[0])(
        name=program.operations[0].name,
        kind="unmodeled_dma",
        inputs=program.operations[0].inputs,
        outputs=program.operations[0].outputs,
        dependencies=program.operations[0].dependencies,
        resource=program.operations[0].resource,
        attributes=program.operations[0].attributes,
    )
    with pytest.raises(ValueError, match="mechanism mismatch"):
        simulate_program(program)

import json

from costmodel import GemmShape, PrecisionSpec, Primitive
from dse import DseRequest, PrismDse, SearchSpace
from generator import HardwareGenerator, build_lut_tables, write_lut_mem


def _result(primitive):
    space = SearchSpace(array_rows=(4,), array_cols=(4,), tile_ks=(8,), dataflows=("output_stationary",), primitives=(primitive,), lut_group_sizes=(1,), lut_digit_bits=(4,), lut_sharing=("global",), hybrid_lut_fractions=(0.5,))
    request = DseRequest(GemmShape(4, 4, 8), PrecisionSpec(4), PrecisionSpec(8), PrecisionSpec(8), search_space=space)
    return request, PrismDse().search(request)


def test_generator_emits_mac_rtl_and_traceable_reports(tmp_path):
    request, result = _result(Primitive.MAC)
    artifacts = HardwareGenerator().generate(request, result, tmp_path, "demo_mac")
    assert artifacts.rtl.exists()
    assert "activation[row] * weight[row][col]" in artifacts.rtl.read_text()
    assert artifacts.lut_init is None
    assert json.loads(artifacts.cost_report.read_text())["primitive"] == "mac"
    assert json.loads(artifacts.manifest.read_text())["rtl_module"] == "demo_mac"


def test_generator_emits_lut_template(tmp_path):
    request, result = _result(Primitive.LUT)
    artifacts = HardwareGenerator().generate(request, result, tmp_path, "demo_lut")
    assert artifacts.lut_init is not None and artifacts.lut_init.exists()
    rtl = artifacts.rtl.read_text()
    assert "$readmemh" in rtl
    assert "LUT_TABLES" in rtl


def test_lut_table_builder_writes_twos_complement_mem(tmp_path):
    tables = build_lut_tables([[3], [-2]], digit_bits=2)
    assert tables[0] == [0, 3, 6, 9]
    assert tables[1] == [0, -2, -4, -6]
    path = write_lut_mem(tmp_path / "weights.mem", tables, output_bits=8)
    assert path.read_text().splitlines()[-1] == "fa"

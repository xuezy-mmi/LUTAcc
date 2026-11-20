from costmodel import Primitive
from dse import SearchSpace
from experiments.llm_sweep import OperatorSpec, parse_precision, run_sweep


def test_precision_parser_and_small_sweep(tmp_path):
    weight, activation, output = parse_precision("W4A16O8")
    assert (weight.bits, activation.bits, output.bits) == (4, 16, 8)
    space = SearchSpace(array_rows=(4,), array_cols=(4,), tile_ks=(8,), dataflows=("output_stationary",), primitives=(Primitive.MAC,), lut_group_sizes=(1,), lut_digit_bits=(4,), lut_sharing=("global",))
    records = run_sweep([OperatorSpec("unit", "decode", "projection", 1, 8, 8)], ["W4A8O8"], tmp_path, space)
    assert records[0]["status"] == "ok"
    assert (tmp_path / "dse_summary.csv").exists()
    assert '"cost_report"' in (tmp_path / "dse_reports.jsonl").read_text()

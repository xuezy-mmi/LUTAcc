import json
from pathlib import Path

from lutacc.cli import main


def test_compile_model_cli_emits_manifest(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            }
        )
    )
    output = tmp_path / "ir"
    assert (
        main(
            [
                "compile-model",
                "--config",
                str(config),
                "--design",
                "oasis",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["program_count"] == 8
    assert all(Path(path).exists() for path in manifest["programs"])


def test_compile_model_omni_emits_full_aw_and_attention_layer(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            }
        )
    )
    output = tmp_path / "omni-ir"
    assert main(
        [
            "compile-model",
            "--config",
            str(config),
            "--design",
            "omni-lut",
            "--decode-tokens",
            "1",
            "--output-dir",
            str(output),
        ]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["program_count"] == 10
    programs = [json.loads(Path(path).read_text()) for path in manifest["programs"]]
    assert sum(item["metadata"]["workload_kind"] == "aw_gemm" for item in programs) == 8
    assert sum(item["metadata"]["workload_kind"] == "attention" for item in programs) == 2


def test_simulate_model_cli_runs_full_omni_stack(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            }
        )
    )
    output = tmp_path / "system.json"
    assert main(
        [
            "simulate-model",
            "--config",
            str(config),
            "--design",
            "omni-lut",
            "--phase",
            "decode",
            "--kv-length",
            "512",
            "--output",
            str(output),
        ]
    ) == 0
    document = json.loads(output.read_text())
    assert document["cycles"] > 0
    assert len(document["layer_breakdown"]) == 5
    assert document["phase"] == "decode"

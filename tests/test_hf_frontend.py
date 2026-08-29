import json

from lutacc.compiler import load_hf_model_spec, model_spec_from_hf_config


def test_llama_style_hf_config_maps_gqa_shape():
    spec = model_spec_from_hf_config(
        {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
        "llama3-8b",
    )
    assert spec.hidden_size == 4096
    assert spec.kv_heads == 8
    assert spec.head_dim == 128


def test_opt_style_hf_config_uses_ffn_dim(tmp_path):
    path = tmp_path / "opt" / "config.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "hidden_size": 4096,
                "ffn_dim": 16384,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            }
        )
    )
    spec = load_hf_model_spec(str(path))
    assert spec.name == "opt"
    assert spec.intermediate_size == 16384
    assert spec.kv_heads == 32


def test_qwen2_style_hf_config_maps_grouped_query_attention():
    spec = model_spec_from_hf_config(
        {
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "head_dim": 128,
        },
        "qwen2-7b",
    )
    assert spec.layers == 28
    assert spec.attention_heads == 28
    assert spec.kv_heads == 4
    assert spec.head_dim == 128

"""Hugging Face config frontend without importing Transformers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..schema import ModelSpec


def model_spec_from_hf_config(
    config: Mapping[str, Any], name: str = "hf-model"
) -> ModelSpec:
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    if hidden % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    intermediate = config.get("intermediate_size", config.get("ffn_dim"))
    if intermediate is None:
        raise KeyError("HF config lacks intermediate_size/ffn_dim")
    kv_heads = int(config.get("num_key_value_heads", heads))
    return ModelSpec(
        name=name,
        layers=int(config["num_hidden_layers"]),
        hidden_size=hidden,
        intermediate_size=int(intermediate),
        attention_heads=heads,
        kv_heads=kv_heads,
        head_dim=int(config.get("head_dim", hidden // heads)),
    )


def load_hf_model_spec(path: str, name: str = "") -> ModelSpec:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("HF config must contain a JSON object")
    return model_spec_from_hf_config(config, name or source.parent.name or source.stem)

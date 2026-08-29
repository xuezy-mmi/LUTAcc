"""Configuration loading with strict scalar/type checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise TypeError("configuration must be a mapping: %s" % source)
    return document


def require_positive(config: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        if key not in config:
            raise KeyError("missing configuration key: %s" % key)
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("configuration %s must be positive" % key)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_hardware_path(design: str) -> Path:
    names = {
        "oasis": "oasis.yaml",
        "omni": "omni_lut_inferred.yaml",
        "omni-lut": "omni_lut_inferred.yaml",
    }
    if design not in names:
        raise KeyError("unknown design: %s" % design)
    return repository_root() / "configs" / "hardware" / names[design]

#!/usr/bin/env python3
"""Run deterministic paper-facing functional and performance experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np

from lutacc.compiler import compile_omni_attention, compile_omni_gemm, compile_oasis_gemm
from lutacc.functional.oasis import oasis_dynamic_outlier_reference, oasis_lut_gemm
from lutacc.functional.omni import omni_attention, stable_softmax
from lutacc.schema import AttentionShape, GemmShape, model_spec
from lutacc.sim.oasis import simulate_oasis_gemm
from lutacc.sim.omni import simulate_omni_attention
from lutacc.sim.omni_system import (
    compare_omni_figlut_long_context,
    simulate_omni_transformer,
)
from lutacc.sim.program import simulate_program
from lutacc.sim.system import (
    compare_oasis_figlut_decode,
    compare_oasis_gpu_decode,
    compare_oasis_lookahead,
)


PAPER_MODELS = [
    model_spec("llama2-7b"),
    model_spec("llama2-13b"),
    model_spec("llama-30b"),
    model_spec("llama2-70b"),
]


def write_json(path: Path, document: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ir_contract() -> Dict[str, Any]:
    programs = {
        "oasis_anchor": compile_oasis_gemm(GemmShape(1, 4096, 4096)),
        "omni_aw": compile_omni_gemm(GemmShape(1, 128, 2048)),
        "omni_decode_attention": compile_omni_attention(
            AttentionShape(1, 8, 1, 2048, 128), "decode"
        ),
    }
    results = {}
    for name, program in programs.items():
        serialized = json.dumps(
            program.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        simulation = simulate_program(program)
        results[name] = {
            "program": program.name,
            "design": program.design,
            "ir_sha256": hashlib.sha256(serialized).hexdigest(),
            "cycles": simulation.cycles,
            "compiler_simulator_contract": simulation.metadata[
                "compiler_simulator_contract"
            ],
        }
    return {"schema_version": 1, "programs": results, "evidence": "measured"}


def environment() -> Dict[str, Any]:
    verilator = None
    if shutil.which("verilator"):
        verilator = subprocess.check_output(["verilator", "--version"], text=True).strip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "verilator": verilator,
    }


def oasis_functional() -> Dict[str, Any]:
    rng = np.random.default_rng(20260829)
    activation = rng.normal(size=(2, 128)).astype(np.float32)
    activation[0, 5] = 32.0
    activation[0, 73] = -29.0
    weights = rng.normal(size=(128, 32)).astype(np.float32)
    result = oasis_lut_gemm(activation, weights, 4, 0.03125)
    reference = oasis_dynamic_outlier_reference(result)
    return {
        "seed": 20260829,
        "shape": [2, 128, 32],
        "maximum_identity_error": float(np.max(np.abs(result.output - reference))),
        "mean_identity_error": float(np.mean(np.abs(result.output - reference))),
        "lut_entries": result.lut_entries,
        "histogram_updates": result.histogram_updates,
    }


def omni_functional() -> Dict[str, Any]:
    rng = np.random.default_rng(20260830)
    query = rng.normal(size=(1, 2, 4, 32)).astype(np.float32)
    key = rng.normal(size=(1, 2, 128, 32)).astype(np.float32)
    value = rng.normal(size=(1, 2, 128, 32)).astype(np.float32)
    compensated = omni_attention(query, key, value, 3.0, 3.0, True)
    raw = omni_attention(query, key, value, 3.0, 3.0, False)
    full = stable_softmax(query @ np.swapaxes(key, -1, -2) / np.sqrt(32.0)) @ value
    fp16_cache_bytes = (key.size + value.size) * 2
    return {
        "seed": 20260830,
        "shape": {"batch": 1, "heads": 2, "query": 4, "kv": 128, "dim": 32},
        "compensated_output_rmse": float(np.sqrt(np.mean(np.square(compensated.output - full)))),
        "uncompensated_output_rmse": float(np.sqrt(np.mean(np.square(raw.output - full)))),
        "quantized_cache_bytes": compensated.key_cache_bytes + compensated.value_cache_bytes,
        "fp16_cache_bytes": fp16_cache_bytes,
        "capacity_reduction": fp16_cache_bytes
        / (compensated.key_cache_bytes + compensated.value_cache_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/results"))
    arguments = parser.parse_args()
    output = arguments.output

    write_json(output / "compiler_simulator_contract.json", ir_contract())

    anchor = simulate_oasis_gemm(GemmShape(1, 4096, 4096), 4, 0.01).to_dict()
    write_json(output / "oasis_pipeline_anchor.json", anchor)

    system = {
        "a4": compare_oasis_figlut_decode(PAPER_MODELS, 4),
        "a3": compare_oasis_figlut_decode(PAPER_MODELS, 3),
    }
    write_json(output / "oasis_single_batch_decode.json", system)
    gpu_system = {
        "a4": compare_oasis_gpu_decode(PAPER_MODELS, 4),
        "a3": compare_oasis_gpu_decode(PAPER_MODELS, 3),
    }
    write_json(output / "oasis_gpu_baselines_decode.json", gpu_system)
    lookahead = {
        "a4": compare_oasis_lookahead(model_spec("llama2-7b"), 4),
        "a3": compare_oasis_lookahead(model_spec("llama2-7b"), 3),
    }
    write_json(output / "oasis_lookahead.json", lookahead)

    outlier_sweep = {
        str(fraction): simulate_oasis_gemm(
            GemmShape(1, 4096, 4096), 4, fraction
        ).to_dict()
        for fraction in (0.005, 0.01, 0.02, 0.05, 0.10)
    }
    write_json(output / "oasis_outlier_sweep.json", outlier_sweep)

    omni_sweep = {"prefill": {}, "decode": {}}
    for context in (128, 512, 2048, 8192, 16384):
        prefill_q = min(context, 512)
        omni_sweep["prefill"][str(context)] = simulate_omni_attention(
            AttentionShape(1, 8, prefill_q, context, 128), "prefill"
        ).to_dict()
        omni_sweep["decode"][str(context)] = simulate_omni_attention(
            AttentionShape(1, 8, 1, context, 128), "decode"
        ).to_dict()
    write_json(output / "omni_context_sweep.json", omni_sweep)
    omni_system_sweep = {"prefill": {}, "decode": {}}
    omni_model = model_spec("llama2-7b")
    for context in (128, 512, 2048, 8192, 16384):
        omni_system_sweep["prefill"][str(context)] = simulate_omni_transformer(
            omni_model,
            "prefill",
            min(context, 512),
            context,
        ).to_dict()
        omni_system_sweep["decode"][str(context)] = simulate_omni_transformer(
            omni_model,
            "decode",
            1,
            context,
        ).to_dict()
    write_json(output / "omni_system_sweep.json", omni_system_sweep)
    omni_energy = compare_omni_figlut_long_context(model_spec("llama2-7b"))
    write_json(output / "omni_energy_proxy.json", omni_energy)

    functional = {"oasis": oasis_functional(), "omni_lut": omni_functional()}
    write_json(output / "functional_equivalence.json", functional)
    write_json(output / "environment.json", environment())

    summary = {
        "oasis_anchor_cycles": anchor["cycles"],
        "oasis_a4_speedup_vs_figlut": system["a4"]["geomean_speedup"],
        "oasis_a3_speedup_vs_figlut": system["a3"]["geomean_speedup"],
        "oasis_a4_energy_efficiency_vs_figlut": system["a4"][
            "geomean_energy_efficiency_improvement"
        ],
        "oasis_a3_energy_efficiency_vs_figlut": system["a3"][
            "geomean_energy_efficiency_improvement"
        ],
        "oasis_functional_max_error": functional["oasis"]["maximum_identity_error"],
        "omni_compensation_improves_rmse": functional["omni_lut"][
            "compensated_output_rmse"
        ]
        <= functional["omni_lut"]["uncompensated_output_rmse"],
        "omni_capacity_reduction": functional["omni_lut"]["capacity_reduction"],
        "omni_energy_efficiency_min": omni_energy["minimum_improvement"],
        "omni_energy_efficiency_max": omni_energy["maximum_improvement"],
        "omni_headline_numeric_validation_available": True,
        "evidence": {
            "oasis_cycle": "calibrated_then_measured",
            "oasis_system": "predicted_vs_inferred_baseline",
            "omni": "predicted_from_inferred_28nm_coefficients",
        },
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

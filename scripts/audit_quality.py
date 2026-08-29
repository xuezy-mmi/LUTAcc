#!/usr/bin/env python3
"""Audit native full-token PPL and zero-shot results, excluding smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_ppl_runs(
    runs: Dict[str, Dict[str, Any]], targets: Dict[str, Any]
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for model_key, run in runs.items():
        target = targets[model_key]
        for mode, path in run["files"].items():
            name = "%s.%s" % (model_key, mode)
            if not path.exists():
                checks.append({"name": name, "available": False, "pass": False})
                continue
            result = load(path)
            measured = float(result["metrics"]["perplexity"])
            expected = float(target[mode])
            relative_error = abs(measured - expected) / expected
            allowed_models = run.get("models", (run.get("model"),))
            full_contract = (
                result["model"] in allowed_models
                and int(result["metrics"]["sequence_length"]) == 2048
                and int(result["metrics"]["evaluated_tokens"])
                >= run["minimum_tokens"]
                and result.get("evidence") == "measured"
            )
            checks.append(
                {
                    "name": name,
                    "available": True,
                    "measured": measured,
                    "target": expected,
                    "relative_error": relative_error,
                    "full_contract": full_contract,
                    "pass": full_contract and relative_error <= 0.10,
                    "path": str(path),
                }
            )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=ROOT / "artifacts" / "results" / "quality",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    reported = load(ROOT / "artifacts" / "targets" / "oasis_reported.json")
    targets = reported["wikitext2_ppl_selected"]
    runs = {
        "opt_6p7b": {
            "model": "facebook/opt-6.7b",
            "minimum_tokens": 280000,
            "files": {
                "fp16": arguments.quality_dir / "opt6.7b_fp16_full.json",
                "oasis_a4": arguments.quality_dir / "opt6.7b_oasis_a4_full.json",
                "oasis_a3": arguments.quality_dir / "opt6.7b_oasis_a3_full.json",
            },
        },
        "mistral_7b": {
            "model": "mistralai/Mistral-7B-v0.1",
            "minimum_tokens": 320000,
            "files": {
                "fp16": arguments.quality_dir / "mistral7b_fp16_full.json",
                "oasis_a4": arguments.quality_dir / "mistral7b_oasis_a4_full.json",
                "oasis_a3": arguments.quality_dir / "mistral7b_oasis_a3_full.json",
            },
        },
    }
    ppl_checks = audit_ppl_runs(runs, targets)
    additional_runs = {
        "llama2_7b": {
            "models": (
                "NousResearch/Llama-2-7b-hf",
                "meta-llama/Llama-2-7b-hf",
            ),
            "minimum_tokens": 270000,
            "files": {
                "fp16": arguments.quality_dir / "llama2_7b_fp16_full.json",
                "oasis_a4": (
                    arguments.quality_dir
                    / "llama2_7b_oasis_a4_zero_refit_full.json"
                ),
                "oasis_a3": arguments.quality_dir / "llama2_7b_oasis_a3_full.json",
            },
        },
        "llama3_8b": {
            "models": (
                "NousResearch/Meta-Llama-3-8B",
                "meta-llama/Meta-Llama-3-8B",
            ),
            "minimum_tokens": 270000,
            "files": {
                "fp16": arguments.quality_dir / "llama3_8b_fp16_full.json",
                "oasis_a4": arguments.quality_dir / "llama3_8b_oasis_a4_full.json",
                "oasis_a3": arguments.quality_dir / "llama3_8b_oasis_a3_full.json",
            },
        },
    }
    additional_ppl_checks = audit_ppl_runs(additional_runs, targets)
    zero_shot_target = reported["zero_shot_mistral_7b"]
    zero_shot_files = {
        "fp16": arguments.quality_dir / "mistral7b_fp16_zeroshot_full.json",
        "oasis_a4": arguments.quality_dir / "mistral7b_oasis_a4_zeroshot_full.json",
        "oasis_a3": arguments.quality_dir / "mistral7b_oasis_a3_zeroshot_full.json",
    }
    tasks = (
        "piqa",
        "arc_easy",
        "arc_challenge",
        "boolq",
        "hellaswag",
        "winogrande",
    )
    zero_shot_checks: List[Dict[str, Any]] = []
    for mode, path in zero_shot_files.items():
        if not path.exists():
            zero_shot_checks.append(
                {
                    "name": "mistral_7b.%s" % mode,
                    "available": False,
                    "pass": False,
                }
            )
            continue
        result = load(path)
        accuracies = result.get("accuracies", {})
        full_contract = (
            result.get("model") == "mistralai/Mistral-7B-v0.1"
            and result.get("mode") == mode.replace("_", "-")
            and result.get("limit") is None
            and result.get("evidence") == "measured"
            and set(tasks).issubset(accuracies)
            and (
                mode == "fp16"
                or (
                    int(result.get("weight_quantized_modules", 0)) > 0
                    and int(result.get("activation_codebooks", 0)) > 0
                    and result.get("calibration_file_sha256")
                    == "b2e620a01e1c5214998562ac7d9ba792a1619e0bbe73cb046a7895fe2eb034af"
                )
            )
        )
        for metric in tasks + ("mean",):
            measured = (
                float(result["mean_accuracy"])
                if metric == "mean"
                else float(accuracies[metric])
            )
            expected = float(zero_shot_target[mode][metric])
            relative_error = abs(measured - expected) / expected
            zero_shot_checks.append(
                {
                    "name": "mistral_7b.%s.%s" % (mode, metric),
                    "available": True,
                    "measured": measured,
                    "target": expected,
                    "relative_error": relative_error,
                    "full_contract": full_contract,
                    "pass": full_contract and relative_error <= 0.10,
                    "path": str(path),
                }
            )

    ppl_pass = all(check["pass"] for check in ppl_checks)
    available_additional_ppl_pass = all(
        check["pass"] for check in additional_ppl_checks if check["available"]
    )
    paper_all_models_ppl_available = all(
        check["available"] for check in ppl_checks + additional_ppl_checks
    )
    paper_all_models_ppl_pass = paper_all_models_ppl_available and all(
        check["pass"] for check in ppl_checks + additional_ppl_checks
    )
    zero_shot_pass = all(check["pass"] for check in zero_shot_checks)
    all_checks = ppl_checks + zero_shot_checks
    document = {
        "models": [run["model"] for run in runs.values()]
        + [run["models"][0] for run in additional_runs.values()],
        "all_available": all(check["available"] for check in all_checks),
        "ppl_pass": ppl_pass,
        "paper_all_models_ppl_available": paper_all_models_ppl_available,
        "paper_all_models_ppl_pass": paper_all_models_ppl_pass,
        "zero_shot_pass": zero_shot_pass,
        "pass": ppl_pass and zero_shot_pass and available_additional_ppl_pass,
        "ppl_checks": ppl_checks,
        "additional_ppl_checks": additional_ppl_checks,
        "zero_shot_checks": zero_shot_checks,
        "smoke_results_excluded": True,
    }
    output = arguments.output or arguments.quality_dir / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

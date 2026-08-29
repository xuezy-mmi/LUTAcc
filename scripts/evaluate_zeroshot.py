#!/usr/bin/env python3
"""Run the six OASIS zero-shot tasks through lm-evaluation-harness v0.4.5."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from lutacc.torch_backend import (
    ActivationCalibration,
    OasisActivationHooks,
    linear_modules,
    quantize_model_weights_oasis,
)


PAPER_TASKS = ["piqa", "arc_easy", "arc_challenge", "boolq", "hellaswag", "winogrande"]


def calibration_texts(
    dataset_name: str, count: int, calibration_file: Path = None
) -> List[str]:
    if calibration_file is not None:
        document = json.loads(calibration_file.read_text(encoding="utf-8"))
        texts = document["samples"]
        if len(texts) < count:
            raise ValueError("calibration file contains fewer samples than requested")
        return texts[:count]
    if dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in dataset:
            text = item.get("text", "").strip()
            if text:
                texts.append(text)
            if len(texts) == count:
                break
        return texts
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [item["text"] for item in dataset if item["text"].strip()][:count]


def install_oasis_hooks(model, tokenizer, arguments):
    weight_metrics = quantize_model_weights_oasis(
        model,
        arguments.maximum_weight_fit_samples,
        arguments.weight_scale_iterations,
        arguments.weight_scale_method,
        arguments.weight_quantizer,
        None,
        arguments.force_zero_weight_centroid,
        arguments.zero_centroid_refit_iterations,
    )
    modules = linear_modules(model)
    collector = ActivationCalibration(
        modules,
        arguments.outlier_fraction,
        weight_energy_fisher_proxy=arguments.weight_energy_fisher_proxy,
        gradient_fisher=arguments.gradient_fisher,
    )
    try:
        for text in calibration_texts(
            arguments.calibration_dataset,
            arguments.calibration_samples,
            arguments.calibration_file,
        ):
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128 if arguments.gradient_fisher else 512,
            ).input_ids.to(model.device)
            if arguments.gradient_fisher:
                model.zero_grad(set_to_none=True)
                output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                output.loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.inference_mode():
                    model(input_ids=input_ids, use_cache=False)
    finally:
        collector.close()
    bits = 4 if arguments.mode == "oasis-a4" else 3
    codebooks = collector.fit(bits)
    hooks = OasisActivationHooks(
        modules,
        codebooks,
        arguments.outlier_fraction,
        arguments.maximum_activation_working_set_elements,
    )
    return hooks, weight_metrics, codebooks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=("fp16", "oasis-a4", "oasis-a3"), default="fp16")
    parser.add_argument("--tasks", default=",".join(PAPER_TASKS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--calibration-dataset", choices=("c4", "wikitext"), default="c4")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--outlier-fraction", type=float, default=0.01)
    parser.add_argument(
        "--maximum-activation-working-set-elements",
        type=int,
        default=4 * 1024 * 1024,
        help="upper bound for each float32 activation-quantization row chunk",
    )
    parser.add_argument("--maximum-weight-fit-samples", type=int, default=262144)
    parser.add_argument("--weight-scale-iterations", type=int, default=0)
    parser.add_argument(
        "--weight-scale-method",
        choices=("maxabs", "rms", "meanabs"),
        default="maxabs",
    )
    parser.add_argument(
        "--weight-quantizer", choices=("kmeans", "uniform"), default="kmeans"
    )
    parser.add_argument("--force-zero-weight-centroid", action="store_true")
    parser.add_argument("--zero-centroid-refit-iterations", type=int, default=0)
    parser.add_argument("--weight-energy-fisher-proxy", action="store_true")
    parser.add_argument("--gradient-fisher", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    arguments = parser.parse_args()
    if arguments.gradient_fisher and arguments.weight_energy_fisher_proxy:
        parser.error("select only one Fisher weighting method")

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.model,
        revision=arguments.revision,
        trust_remote_code=arguments.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model,
        revision=arguments.revision,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=arguments.trust_remote_code,
    )
    model.eval()
    hooks = None
    weight_metrics: Dict[str, Any] = {}
    codebooks: Dict[str, Any] = {}
    if arguments.mode != "fp16":
        hooks, weight_metrics, codebooks = install_oasis_hooks(model, tokenizer, arguments)

    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=arguments.batch_size,
        max_length=2048,
    )
    start = time.time()
    try:
        raw = simple_evaluate(
            model=wrapped,
            tasks=[task for task in arguments.tasks.split(",") if task],
            num_fewshot=0,
            batch_size=arguments.batch_size,
            limit=arguments.limit,
            bootstrap_iters=1000,
            log_samples=False,
        )
    finally:
        if hooks is not None:
            hooks.close()
    elapsed = time.time() - start
    task_results = raw["results"]
    accuracies = {}
    for task, metrics in task_results.items():
        selected = None
        for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
            if key in metrics:
                selected = float(metrics[key])
                break
        accuracies[task] = selected
    valid = [value for value in accuracies.values() if value is not None]
    document = {
        "model": arguments.model,
        "model_revision": arguments.revision,
        "mode": arguments.mode,
        "tasks": task_results,
        "accuracies": accuracies,
        "mean_accuracy": sum(valid) / len(valid),
        "sequence_length": 2048,
        "elapsed_seconds": elapsed,
        "limit": arguments.limit,
        "weight_quantized_modules": len(weight_metrics),
        "activation_codebooks": len(codebooks),
        "calibration_dataset": arguments.calibration_dataset,
        "calibration_samples": arguments.calibration_samples,
        "calibration_file": (
            str(arguments.calibration_file) if arguments.calibration_file else None
        ),
        "calibration_file_sha256": (
            hashlib.sha256(arguments.calibration_file.read_bytes()).hexdigest()
            if arguments.calibration_file
            else None
        ),
        "gradient_fisher": arguments.gradient_fisher,
        "weight_energy_fisher_proxy": arguments.weight_energy_fisher_proxy,
        "weight_quantizer": arguments.weight_quantizer,
        "force_zero_weight_centroid": arguments.force_zero_weight_centroid,
        "zero_centroid_refit_iterations": arguments.zero_centroid_refit_iterations,
        "outlier_fraction": arguments.outlier_fraction,
        "maximum_activation_working_set_elements": (
            arguments.maximum_activation_working_set_elements
        ),
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "torch_version": torch.__version__,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "evidence": "measured" if arguments.limit is None else "measured_smoke",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

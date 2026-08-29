#!/usr/bin/env python3
"""Evaluate FP16 or OASIS W4A4/W4A3 perplexity on a real HF causal LM."""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lutacc.torch_backend import (
    ActivationCalibration,
    InputHessianCalibration,
    OasisActivationHooks,
    linear_modules,
    quantize_model_weights_oasis,
)


def collect_calibration_texts(
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
        result = []
        for item in dataset:
            text = item.get("text", "").strip()
            if text:
                result.append(text)
            if len(result) >= count:
                break
        return result
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [item["text"] for item in dataset if item["text"].strip()][:count]


def calibrate_activations(
    model,
    tokenizer,
    texts: Iterable[str],
    sequence_length: int,
    bits: int,
    outlier_fraction: float,
    weight_energy_fisher_proxy: bool,
    gradient_fisher: bool,
):
    modules = linear_modules(model)
    collector = ActivationCalibration(
        modules,
        outlier_fraction,
        weight_energy_fisher_proxy=weight_energy_fisher_proxy,
        gradient_fisher=gradient_fisher,
    )
    try:
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length,
            )
            input_ids = encoded.input_ids.to(model.device)
            if gradient_fisher:
                model.zero_grad(set_to_none=True)
                output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                output.loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.inference_mode():
                    model(input_ids=input_ids, use_cache=False)
    finally:
        collector.close()
    return collector.fit(bits)


def calibrate_weight_hessian(model, tokenizer, texts: Iterable[str], sequence_length: int):
    collector = InputHessianCalibration(linear_modules(model))
    try:
        for text in texts:
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length,
            ).input_ids.to(model.device)
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False)
    finally:
        collector.close()
    return collector.result()


def evaluate_wikitext2(model, tokenizer, sequence_length: int, maximum_chunks: int) -> Dict[str, Any]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(item["text"] for item in dataset if item["text"].strip())
    tokens = tokenizer(text, return_tensors="pt").input_ids
    chunks = min(tokens.shape[1] // sequence_length, maximum_chunks)
    if chunks < 1:
        raise RuntimeError("WikiText-2 token stream is shorter than sequence length")
    total_nll = 0.0
    total_tokens = 0
    start = time.time()
    for chunk in range(chunks):
        begin = chunk * sequence_length
        input_ids = tokens[:, begin : begin + sequence_length].to(model.device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        predicted = input_ids.numel() - input_ids.shape[0]
        total_nll += float(output.loss) * predicted
        total_tokens += predicted
    elapsed = time.time() - start
    return {
        "perplexity": math.exp(total_nll / total_tokens),
        "evaluated_tokens": total_tokens,
        "chunks": chunks,
        "sequence_length": sequence_length,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=("fp16", "oasis-a4", "oasis-a3"), default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-eval-chunks", type=int, default=8)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--calibration-dataset", choices=("c4", "wikitext"), default="c4")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--outlier-fraction", type=float, default=0.01)
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
    parser.add_argument(
        "--weight-input-hessian",
        action="store_true",
        help="weight W4 K-Means by calibrated per-input-channel E[x^2]",
    )
    parser.add_argument(
        "--force-zero-weight-centroid",
        action="store_true",
        help="pin the nearest W4 K-Means centroid to exact zero",
    )
    parser.add_argument("--zero-centroid-refit-iterations", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-weight-quantization", action="store_true")
    parser.add_argument("--skip-activation-quantization", action="store_true")
    parser.add_argument(
        "--weight-energy-fisher-proxy",
        action="store_true",
        help="weight activation samples by downstream column energy; off by default",
    )
    parser.add_argument(
        "--gradient-fisher",
        action="store_true",
        help="use squared loss gradients as weighted-K-Means Fisher information",
    )
    arguments = parser.parse_args()
    if arguments.gradient_fisher and arguments.weight_energy_fisher_proxy:
        parser.error("select only one Fisher weighting method")

    total_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.model,
        revision=arguments.revision,
        trust_remote_code=arguments.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model,
        revision=arguments.revision,
        torch_dtype=torch.float16,
        device_map=arguments.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=arguments.trust_remote_code,
    )
    model.eval()
    load_seconds = time.time() - load_start
    weight_metrics: Dict[str, Any] = {}
    codebooks = {}
    hooks = None
    weight_quantization_seconds = 0.0
    weight_hessian_calibration_seconds = 0.0
    activation_calibration_seconds = 0.0
    if arguments.mode != "fp16":
        texts = None
        if arguments.weight_input_hessian or not arguments.skip_activation_quantization:
            texts = collect_calibration_texts(
                arguments.calibration_dataset,
                arguments.calibration_samples,
                arguments.calibration_file,
            )
        input_importance = None
        if arguments.weight_input_hessian:
            hessian_start = time.time()
            input_importance = calibrate_weight_hessian(
                model, tokenizer, texts, min(arguments.sequence_length, 512)
            )
            weight_hessian_calibration_seconds = time.time() - hessian_start
        if not arguments.skip_weight_quantization:
            quantization_start = time.time()
            weight_metrics = quantize_model_weights_oasis(
                model,
                arguments.maximum_weight_fit_samples,
                arguments.weight_scale_iterations,
                arguments.weight_scale_method,
                arguments.weight_quantizer,
                input_importance,
                arguments.force_zero_weight_centroid,
                arguments.zero_centroid_refit_iterations,
            )
            weight_quantization_seconds = time.time() - quantization_start
        bits = 4 if arguments.mode == "oasis-a4" else 3
        if not arguments.skip_activation_quantization:
            calibration_start = time.time()
            codebooks = calibrate_activations(
                model,
                tokenizer,
                texts,
                min(arguments.sequence_length, 128 if arguments.gradient_fisher else 512),
                bits,
                arguments.outlier_fraction,
                arguments.weight_energy_fisher_proxy,
                arguments.gradient_fisher,
            )
            hooks = OasisActivationHooks(
                linear_modules(model), codebooks, arguments.outlier_fraction
            )
            activation_calibration_seconds = time.time() - calibration_start
    try:
        metrics = evaluate_wikitext2(
            model, tokenizer, arguments.sequence_length, arguments.max_eval_chunks
        )
    finally:
        if hooks is not None:
            hooks.close()
    document = {
        "model": arguments.model,
        "model_revision": arguments.revision,
        "mode": arguments.mode,
        "metrics": metrics,
        "weight_quantized_modules": len(weight_metrics),
        "skip_weight_quantization": arguments.skip_weight_quantization,
        "skip_activation_quantization": arguments.skip_activation_quantization,
        "activation_codebooks": len(codebooks),
        "mean_weight_rmse": (
            sum(item["rmse"] for item in weight_metrics.values()) / len(weight_metrics)
            if weight_metrics
            else 0.0
        ),
        "weight_scale_iterations": arguments.weight_scale_iterations,
        "weight_scale_method": arguments.weight_scale_method,
        "weight_quantizer": arguments.weight_quantizer,
        "weight_input_hessian": arguments.weight_input_hessian,
        "force_zero_weight_centroid": arguments.force_zero_weight_centroid,
        "zero_centroid_refit_iterations": arguments.zero_centroid_refit_iterations,
        "calibration_samples": arguments.calibration_samples,
        "calibration_dataset": arguments.calibration_dataset,
        "calibration_file": (
            str(arguments.calibration_file) if arguments.calibration_file else None
        ),
        "calibration_file_sha256": (
            hashlib.sha256(arguments.calibration_file.read_bytes()).hexdigest()
            if arguments.calibration_file
            else None
        ),
        "outlier_fraction": arguments.outlier_fraction,
        "weight_energy_fisher_proxy": arguments.weight_energy_fisher_proxy,
        "gradient_fisher": arguments.gradient_fisher,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "evidence": "measured",
        "timing": {
            "model_load_seconds": load_seconds,
            "weight_quantization_seconds": weight_quantization_seconds,
            "weight_hessian_calibration_seconds": weight_hessian_calibration_seconds,
            "activation_calibration_seconds": activation_calibration_seconds,
            "evaluation_seconds": metrics["elapsed_seconds"],
            "total_seconds": time.time() - total_start,
        },
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

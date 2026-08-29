"""Optional PyTorch backend for real-model OASIS quality evaluation.

This module is intentionally not imported by :mod:`lutacc` so the compiler and
cycle simulator remain lightweight. `scripts/bootstrap_ml.sh` provides its
pinned environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import torch
import torch.nn as nn


@dataclass
class TorchCodebook:
    centroids: torch.Tensor

    @property
    def boundaries(self) -> torch.Tensor:
        return (self.centroids[:-1] + self.centroids[1:]) * 0.5

    def to(self, device: torch.device) -> "TorchCodebook":
        return TorchCodebook(self.centroids.to(device=device))


def uniform_sample_positions(
    length: int, count: int, device: torch.device
) -> torch.Tensor:
    """Even integer positions without float32 rounding past large tensors."""

    if length <= 0 or count <= 0:
        raise ValueError("sample length and count must be positive")
    return torch.linspace(
        0,
        length - 1,
        min(length, count),
        device=device,
        dtype=torch.float64,
    ).long().clamp_max(length - 1)


def fit_torch_kmeans_1d(
    values: torch.Tensor,
    levels: int,
    iterations: int = 32,
    maximum_samples: int = 262144,
    sample_weights: torch.Tensor = None,
) -> TorchCodebook:
    data = values.detach().reshape(-1).float()
    finite = torch.isfinite(data)
    data = data[finite]
    weights = (
        torch.ones_like(data)
        if sample_weights is None
        else sample_weights.detach().reshape(-1).float()[finite]
    )
    if torch.any(weights < 0) or not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("K-Means sample weights must be finite and non-negative")
    if data.numel() < levels:
        raise ValueError("not enough finite values to fit codebook")
    if data.numel() > maximum_samples:
        positions = uniform_sample_positions(
            data.numel(), maximum_samples, data.device
        )
        data = data[positions]
        weights = weights[positions]
    # Unweighted quantile seeding prevents high-Fisher channels from collapsing
    # several initial clusters onto the same common value. Lloyd updates below
    # still optimize the weighted objective used by OASIS.
    quantiles = (
        torch.arange(levels, device=data.device, dtype=torch.float32) + 0.5
    ) / levels
    centroids = torch.quantile(data, quantiles).sort().values
    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        assignment = torch.bucketize(data, boundaries)
        sums = torch.zeros(levels, device=data.device, dtype=torch.float32)
        counts = torch.zeros(levels, device=data.device, dtype=torch.float32)
        sums.scatter_add_(0, assignment, data * weights)
        counts.scatter_add_(0, assignment, weights)
        updated = torch.where(counts > 0, sums / counts.clamp_min(1e-30), centroids)
        updated = updated.sort().values
        if torch.max(torch.abs(updated - centroids)).item() < 1e-6:
            centroids = updated
            break
        centroids = updated
    return TorchCodebook(centroids.detach().cpu())


def refit_torch_kmeans_with_fixed_zero(
    values: torch.Tensor,
    codebook: TorchCodebook,
    sample_weights: torch.Tensor = None,
    iterations: int = 8,
) -> TorchCodebook:
    """Run constrained Lloyd updates while keeping one centroid exactly zero."""

    if iterations < 0:
        raise ValueError("fixed-zero refit iterations cannot be negative")
    data = values.detach().reshape(-1).float()
    finite = torch.isfinite(data)
    data = data[finite]
    weights = (
        torch.ones_like(data)
        if sample_weights is None
        else sample_weights.detach().reshape(-1).float()[finite]
    )
    if data.numel() < codebook.centroids.numel() or weights.sum() <= 0:
        raise ValueError("fixed-zero refit requires enough weighted samples")
    centroids = codebook.centroids.to(data.device).clone().sort().values
    centroids[torch.argmin(torch.abs(centroids))] = 0.0
    levels = centroids.numel()
    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        assignment = torch.bucketize(data, boundaries)
        sums = torch.zeros(levels, device=data.device, dtype=torch.float32)
        counts = torch.zeros(levels, device=data.device, dtype=torch.float32)
        sums.scatter_add_(0, assignment, data * weights)
        counts.scatter_add_(0, assignment, weights)
        updated = torch.where(counts > 0, sums / counts.clamp_min(1e-30), centroids)
        updated[torch.argmin(torch.abs(centroids))] = 0.0
        updated = updated.sort().values
        if torch.max(torch.abs(updated - centroids)).item() < 1e-6:
            centroids = updated
            break
        centroids = updated
    return TorchCodebook(centroids.detach().cpu())


def quantize_linear_weight_oasis(
    module: nn.Linear,
    bits: int = 4,
    row_chunk: int = 256,
    maximum_fit_samples: int = 262144,
    scale_iterations: int = 0,
    scale_method: str = "maxabs",
    input_importance: torch.Tensor = None,
    force_zero_centroid: bool = False,
    zero_centroid_refit_iterations: int = 0,
) -> Dict[str, Any]:
    """Quantize one [out,in] weight with global centroids/per-output scale."""

    if bits != 4:
        raise ValueError("OASIS weights are fixed at four bits")
    if scale_iterations < 0:
        raise ValueError("scale_iterations cannot be negative")
    if zero_centroid_refit_iterations < 0:
        raise ValueError("zero-centroid refit iterations cannot be negative")
    if scale_method not in {"maxabs", "rms", "meanabs"}:
        raise ValueError("unsupported weight scale method")
    weight = module.weight.data
    if weight.device.type == "meta":
        raise ValueError("cannot quantize a meta/offloaded weight")
    output_features, input_features = weight.shape
    scales = torch.empty(output_features, device=weight.device, dtype=torch.float32)
    for begin in range(0, output_features, row_chunk):
        end = min(begin + row_chunk, output_features)
        rows = weight[begin:end].float()
        if scale_method == "maxabs":
            local_scale = rows.abs().amax(dim=1)
        elif scale_method == "meanabs":
            local_scale = rows.abs().mean(dim=1)
        else:
            local_scale = rows.square().mean(dim=1).sqrt()
        scales[begin:end] = local_scale.clamp_min(1e-12)

    sample_count = min(maximum_fit_samples, weight.numel())
    flat_positions = uniform_sample_positions(
        weight.numel(), sample_count, weight.device
    )
    sample_rows = torch.div(flat_positions, input_features, rounding_mode="floor")
    sample_columns = flat_positions.remainder(input_features)
    importance = (
        torch.ones(input_features, device=weight.device, dtype=torch.float32)
        if input_importance is None
        else input_importance.to(device=weight.device, dtype=torch.float32)
    )
    if importance.numel() != input_features or torch.any(importance < 0):
        raise ValueError("input importance must match Linear input features")
    importance = importance.reshape(-1).clamp_min(1e-12)
    sample_importance = importance[sample_columns]
    codebook = None
    for iteration in range(scale_iterations + 1):
        normalized_sample = (
            weight[sample_rows, sample_columns].float() / scales[sample_rows]
        )
        codebook = fit_torch_kmeans_1d(
            normalized_sample,
            1 << bits,
            maximum_samples=sample_count,
            sample_weights=sample_importance,
        )
        centroids = codebook.centroids.to(weight.device)
        if force_zero_centroid:
            centroids[torch.argmin(torch.abs(centroids))] = 0.0
            codebook = TorchCodebook(centroids.detach().cpu())
            if zero_centroid_refit_iterations:
                codebook = refit_torch_kmeans_with_fixed_zero(
                    normalized_sample,
                    codebook,
                    sample_importance,
                    zero_centroid_refit_iterations,
                )
                centroids = codebook.centroids.to(weight.device)
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        if iteration == scale_iterations:
            break
        # With centroid assignments fixed, this is the exact least-squares
        # per-output-channel scale update. Alternating it with global K-Means
        # reconstructs the paper's learned-codebook + channel-scale objective.
        for begin in range(0, output_features, row_chunk):
            end = min(begin + row_chunk, output_features)
            original = weight[begin:end].float()
            normalized = original / scales[begin:end, None]
            indices = torch.bucketize(normalized, boundaries)
            codes = centroids[indices]
            row_importance = importance.reshape(1, -1)
            numerator = torch.sum(original * codes * row_importance, dim=1)
            denominator = torch.sum(
                codes.square() * row_importance, dim=1
            ).clamp_min(1e-12)
            scales[begin:end] = (numerator / denominator).clamp_min(1e-12)
    squared_error = 0.0
    element_count = 0
    for begin in range(0, output_features, row_chunk):
        end = min(begin + row_chunk, output_features)
        original = weight[begin:end].float()
        normalized = original / scales[begin:end, None]
        indices = torch.bucketize(normalized, boundaries)
        reconstructed = centroids[indices] * scales[begin:end, None]
        squared_error += torch.sum(torch.square(original - reconstructed)).item()
        element_count += original.numel()
        weight[begin:end].copy_(reconstructed.to(dtype=weight.dtype))
    return {
        "output_features": output_features,
        "input_features": input_features,
        "bits": bits,
        "scale_iterations": scale_iterations,
        "scale_method": scale_method,
        "input_hessian_weighted": input_importance is not None,
        "force_zero_centroid": force_zero_centroid,
        "zero_centroid_refit_iterations": zero_centroid_refit_iterations,
        "rmse": math.sqrt(squared_error / max(element_count, 1)),
        "centroids": codebook.centroids.tolist(),
    }


def quantize_linear_weight_uniform(module: nn.Linear) -> Dict[str, Any]:
    """Symmetric W4 RTN diagnostic baseline with per-output-channel scale."""

    weight = module.weight.data
    original = weight.float()
    scales = original.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0
    codes = torch.round(original / scales).clamp(-8, 7)
    reconstructed = codes * scales
    rmse = torch.mean(torch.square(original - reconstructed)).sqrt().item()
    weight.copy_(reconstructed.to(dtype=weight.dtype))
    return {
        "output_features": weight.shape[0],
        "input_features": weight.shape[1],
        "bits": 4,
        "rmse": rmse,
        "scale_method": "maxabs_over_7",
        "quantizer": "uniform_rtn",
    }


def linear_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.endswith("lm_head")
    ]


def quantize_model_weights_oasis(
    model: nn.Module,
    maximum_fit_samples: int = 262144,
    scale_iterations: int = 0,
    scale_method: str = "maxabs",
    quantizer: str = "kmeans",
    input_importance: Mapping[str, torch.Tensor] = None,
    force_zero_centroid: bool = False,
    zero_centroid_refit_iterations: int = 0,
) -> Dict[str, Any]:
    if quantizer not in {"kmeans", "uniform"}:
        raise ValueError("weight quantizer must be kmeans or uniform")
    results: Dict[str, Any] = {}
    for name, module in linear_modules(model):
        results[name] = (
            quantize_linear_weight_oasis(
                module,
                maximum_fit_samples=maximum_fit_samples,
                scale_iterations=scale_iterations,
                scale_method=scale_method,
                input_importance=(input_importance or {}).get(name),
                force_zero_centroid=force_zero_centroid,
                zero_centroid_refit_iterations=zero_centroid_refit_iterations,
            )
            if quantizer == "kmeans"
            else quantize_linear_weight_uniform(module)
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


class InputHessianCalibration:
    """Collect diagonal E[x^2] input Hessian proxies for every Linear."""

    def __init__(self, modules: Iterable[Tuple[str, nn.Linear]]):
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self.handles = []
        for name, module in modules:
            self.sums[name] = torch.zeros(module.in_features, dtype=torch.float64)
            self.counts[name] = 0
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        def collect(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
            self.sums[name] += rows.square().sum(dim=0).cpu()
            self.counts[name] += rows.shape[0]
            return None

        return collect

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def result(self) -> Dict[str, torch.Tensor]:
        return {
            name: self.sums[name].float() / max(self.counts[name], 1)
            for name in self.sums
            if self.counts[name] > 0
        }


def _normalized_inlier_sample(
    activation: torch.Tensor,
    outlier_fraction: float,
    maximum_values: int,
    channel_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = activation.detach().reshape(-1, activation.shape[-1]).float()
    k_each = int(rows.shape[-1] * outlier_fraction * 0.5)
    mask = torch.zeros_like(rows, dtype=torch.bool)
    if k_each:
        high = torch.topk(rows, k_each, dim=-1, largest=True, sorted=False).indices
        low = torch.topk(rows, k_each, dim=-1, largest=False, sorted=False).indices
        mask.scatter_(1, high, True)
        mask.scatter_(1, low, True)
    scale = rows.abs().masked_fill(mask, 0).amax(dim=-1, keepdim=True).clamp_min(1e-12)
    values = (rows / scale)[~mask]
    fisher = channel_weights.to(device=rows.device, dtype=torch.float32)
    fisher = fisher.reshape(1, -1).expand_as(rows)[~mask]
    if values.numel() > maximum_values:
        positions = uniform_sample_positions(
            values.numel(), maximum_values, values.device
        )
        values = values[positions]
        fisher = fisher[positions]
    return values.detach().cpu(), fisher.detach().cpu()


class ActivationCalibration:
    def __init__(
        self,
        modules: Iterable[Tuple[str, nn.Linear]],
        outlier_fraction: float = 0.01,
        maximum_values_per_module: int = 32768,
        weight_energy_fisher_proxy: bool = False,
        gradient_fisher: bool = False,
    ):
        self.outlier_fraction = outlier_fraction
        self.maximum_values_per_module = maximum_values_per_module
        self.weight_energy_fisher_proxy = weight_energy_fisher_proxy
        self.gradient_fisher = gradient_fisher
        if self.gradient_fisher and self.weight_energy_fisher_proxy:
            raise ValueError("select only one Fisher weighting method")
        self.samples: MutableMapping[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {
            name: [] for name, _ in modules
        }
        self.handles = []
        for name, module in modules:
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        def collect(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            collected = sum(item[0].numel() for item in self.samples[name])
            remaining = self.maximum_values_per_module - collected
            if remaining > 0:
                if self.gradient_fisher:
                    activation = inputs[0]
                    rows = activation.detach().reshape(
                        -1, activation.shape[-1]
                    ).float()
                    k_each = int(rows.shape[-1] * self.outlier_fraction * 0.5)
                    mask = torch.zeros_like(rows, dtype=torch.bool)
                    if k_each:
                        high = torch.topk(
                            rows, k_each, dim=-1, largest=True, sorted=False
                        ).indices
                        low = torch.topk(
                            rows, k_each, dim=-1, largest=False, sorted=False
                        ).indices
                        mask.scatter_(1, high, True)
                        mask.scatter_(1, low, True)
                    scale = rows.abs().masked_fill(mask, 0).amax(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-12)
                    normalized = (rows / scale).reshape(-1)
                    positions = torch.nonzero(
                        (~mask).reshape(-1), as_tuple=False
                    ).reshape(-1)
                    if positions.numel() > remaining:
                        choose = uniform_sample_positions(
                            positions.numel(), remaining, positions.device
                        )
                        positions = positions[choose]
                    values = normalized[positions].detach().cpu()

                    def capture_gradient(gradient: torch.Tensor) -> None:
                        fisher = gradient.detach().reshape(-1)[positions]
                        fisher = fisher.float().square().clamp_min(1e-12).cpu()
                        self.samples[name].append((values, fisher))

                    if activation.requires_grad:
                        activation.register_hook(capture_gradient)
                    else:
                        self.samples[name].append((values, torch.ones_like(values)))
                    return None
                channel_weights = (
                    module.weight.detach().float().square().mean(dim=0).clamp_min(1e-12)
                    if self.weight_energy_fisher_proxy
                    else torch.ones(module.weight.shape[1], device=module.weight.device)
                )
                values, weights = _normalized_inlier_sample(
                    inputs[0], self.outlier_fraction, remaining, channel_weights
                )
                self.samples[name].append((values[:remaining], weights[:remaining]))
            return None

        return collect

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def fit(self, bits: int) -> Dict[str, TorchCodebook]:
        codebooks = {}
        for name, samples in self.samples.items():
            if not samples:
                continue
            values = torch.cat([sample[0] for sample in samples])
            weights = torch.cat([sample[1] for sample in samples])
            codebooks[name] = fit_torch_kmeans_1d(
                values, 1 << bits, sample_weights=weights
            )
        return codebooks


class OasisActivationHooks:
    def __init__(
        self,
        modules: Iterable[Tuple[str, nn.Linear]],
        codebooks: Mapping[str, TorchCodebook],
        outlier_fraction: float = 0.01,
        maximum_working_set_elements: int = 4 * 1024 * 1024,
    ):
        if maximum_working_set_elements <= 0:
            raise ValueError("maximum working-set elements must be positive")
        self.outlier_fraction = outlier_fraction
        self.maximum_working_set_elements = maximum_working_set_elements
        self.handles = []
        self.device_codebooks: Dict[Tuple[str, str], TorchCodebook] = {}
        for name, module in modules:
            if name in codebooks:
                self.handles.append(
                    module.register_forward_pre_hook(self._hook(name, codebooks[name]))
                )

    def _hook(self, name: str, codebook: TorchCodebook):
        def quantize(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            activation = inputs[0]
            original_shape = activation.shape
            flat = activation.reshape(-1, original_shape[-1])
            quantized = torch.empty_like(flat)
            row_chunk = max(
                1, self.maximum_working_set_elements // original_shape[-1]
            )
            cache_key = (name, str(activation.device))
            if cache_key not in self.device_codebooks:
                self.device_codebooks[cache_key] = codebook.to(activation.device)
            local = self.device_codebooks[cache_key]
            boundaries = local.boundaries
            k_each = int(original_shape[-1] * self.outlier_fraction * 0.5)
            # lm-eval can present [batch, sequence, hidden] tensors with hundreds
            # of millions of elements.  Quantizing that tensor in one float32
            # expression creates several equally large temporaries and exceeds a
            # 24-GiB GPU even though the model itself fits.  Each row is
            # independent in OASIS, so bounded row chunks are bit-for-bit the
            # same operation while keeping the hook's transient memory bounded.
            for begin in range(0, flat.shape[0], row_chunk):
                end = min(begin + row_chunk, flat.shape[0])
                rows = flat[begin:end].float()
                mask = torch.zeros_like(rows, dtype=torch.bool)
                if k_each:
                    high = torch.topk(
                        rows, k_each, dim=-1, largest=True, sorted=False
                    ).indices
                    low = torch.topk(
                        rows, k_each, dim=-1, largest=False, sorted=False
                    ).indices
                    mask.scatter_(1, high, True)
                    mask.scatter_(1, low, True)
                scale = rows.abs().masked_fill(mask, 0).amax(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                indices = torch.bucketize(rows / scale, boundaries)
                reconstructed = local.centroids[indices] * scale
                reconstructed = torch.where(mask, rows, reconstructed)
                quantized[begin:end].copy_(reconstructed.to(dtype=activation.dtype))
            return (quantized.reshape(original_shape),) + tuple(inputs[1:])

        return quantize

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

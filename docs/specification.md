# Reconstructed specification

## Shared contract

The compiler emits a JSON DAG of typed tensors and operations. `simulate-ir`
round-trips that JSON, rejects missing or unknown mechanism nodes, and dispatches
the exact serialized shapes and precisions into the cycle model. Every run
reports cycles, stage intervals, conventional operation count, DRAM/SRAM bytes,
energy, throughput, and provenance metadata.

## OASIS

Paper-backed configuration (arXiv:2507.23035v4):

- 28 nm, 500 MHz, 15.31 mm², 9.66 W.
- 16 PE Lines; each has 4096 4b+4b Concat Units, a 2 KiB weight-index
  buffer, 32 16-input index counters, a dequantizer, one 32-input FP16 MAC
  tree, and eight FP16 compensation MACs.
- Chip-level 64 KiB output buffer, 16 KiB activation-index buffer, 2 KiB LUT,
  four clustering units, and 273 16-input Orizuru units.
- Weight K-Means uses a global normalized codebook with per-output-channel
  scale. Activation K-Means uses an offline normalized codebook with per-token
  scale. Top 0.5% and bottom 0.5% activations are protected for the default 1%
  budget.
- Main branch concatenates activation and weight indices, forms a histogram,
  and performs a weighted sum over the 128/256-entry Cartesian-product LUT.
  The outlier branch computes `(a - aq) @ dequant(W)` only at selected channels.

For the paper's 1x4096x4096 W4A4/1% point, the model derives:

| Component | Cycles |
|---|---:|
| Activation quantization | 2048 |
| Concat stream | 256 |
| Index distribution | 1024 |
| Reduction pipeline | 2304 |
| Orizuru detection | 1344 |
| Error compensation | 1536 |
| Merge | 256 |
| Critical total | 2560 |

The first six components overlap as streaming main/outlier branches; merge
starts after both branches. The cycle formulas are shape dependent and the
2,560 total is not stored in simulator code.

The open A100/QuaRot comparisons use a declared single-batch roofline. A100
peak bandwidth, tensor throughput, and TDP come from NVIDIA's public 80-GB SXM
datasheet; effective bandwidth/power and the QuaRot utilization proxy are kept
in `configs/calibration/gpu_decode_roofline_v1.yaml`. They train on the public
headline geomeans and are labeled `calibrated_roofline_not_native_measurement`.

## Omni-LUT

Publicly supported claims are AW- and AA-GEMM LUT execution, offline+online
hardware-aware KV quantization, quantization compensation during LUT creation,
and a phase-adaptive hybrid-stationary systolic array. The supplied project brief
adds AS-Bit, BQU, LGU, a 32x4 LUT-PE array, Prefill-WS, and Decode-OS-V.

The open reconstruction implements:

- AS-Bit as exact-budget 2/3/4-bit allocation using marginal reduction in a
  second-moment quantization-error proxy.
- Key allocation per channel and Value allocation per token.
- BQU symmetric variable-bit encoding plus per-code mean-residual calibration.
- LGU tables that fold that residual compensation into each centroid product.
- AW-GEMM product tables formed from FP16 activations and W4 learned centroids,
  followed by real weight-index gathers and per-output scaling.
- QK and PV as actual indexed table accumulation, not dequantize-then-GEMM.
- A 32x4 array event model with phase-specific mapping and explicitly inferred
  timing constants in `configs/hardware/omni_lut_inferred.yaml`.
- A full transformer-layer aggregate containing QKV/O/MLP AW-GEMMs plus QK,
  softmax, and PV; decode switches to tiled DRAM KV reads when the packed cache
  exceeds the configured 2 MiB KV SRAM.

These Omni-LUT choices remain `inferred` until a full author source or public
paper artifact supplies the missing constants.

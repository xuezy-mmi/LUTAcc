# PRISM Cost Model

## Purpose

`costmodel` is the first-stage analytical estimator for PRISM. It compares MAC, LUT, and hybrid GEMM implementations before expensive RTL generation. It is intentionally dependency-free and conservative: the default coefficients support relative design-space ranking only, not process-specific PPA claims.

## Inputs

- `GemmShape(m, n, k)`: output dimensions and reduction length.
- `PrecisionSpec(bits, kind, signed)`: weight, activation, and accumulator format. `kind` is `int` or `fp`.
- `ArrayConfig(rows, cols, tile_k, dataflow)`: compute-array dimensions and reduction tile.
- `Primitive`: `mac`, `lut`, or `hybrid`.
- `LutConfig(group_size, digit_bits, sharing, max_index_bits)`: LUT decomposition and placement policy.

For a hybrid design, pass `primitive=Primitive.HYBRID` and select `lut_fraction` in `[0, 1]`. In the current abstraction this is a fraction of output work assigned to LUT tiles. The DSE module will later replace it with an explicit tile assignment map.

## Model equations

MAC operations are `M * N * K`. A digit-decomposed LUT uses:

```text
groups              = ceil(K / group_size)
digits              = ceil(A_bits / digit_bits)
tables/output       = groups * digits
index_bits          = group_size * min(digit_bits, A_bits)
entries/table       = 2 ^ index_bits
table_output_bits   = W_bits + min(digit_bits, A_bits) + ceil(log2(group_size))
```

The model rejects tables whose index width exceeds `max_index_bits`; this avoids silently accepting impractical W4A16 tables. LUT storage includes replication controlled by `sharing`:

- `per_pe`: one copy per PE;
- `per_tile`: one copy per logical 4x4 sub-array;
- `global`: one shared copy.

Area includes compute storage/logic plus a first-order per-PE interconnect term. Energy includes arithmetic or lookup energy and a first-order movement term. Latency is tile traversal plus reduction overhead. It is a throughput-oriented approximation and does not yet model memory port contention, routing congestion, or detailed pipeline stages.

## Calibration workflow

1. Synthesize parameterized MAC, accumulator, LUT/ROM, SRAM, and interconnect microblocks for the target technology.
2. Measure area, timing, and switching energy over every intended bitwidth.
3. Fit `TechnologyConfig` coefficients, adding terms when residuals show systematic errors.
4. Hold out representative array sizes and GEMM shapes; report mean and worst-case relative error.
5. Use calibrated coefficients in DSE, then re-synthesize the selected Pareto points.

## Example

```python
from costmodel import ArrayConfig, GemmShape, LutConfig, Primitive, PrecisionSpec, PrismCostModel

estimate = PrismCostModel().estimate(
    shape=GemmShape(32, 4096, 4096),
    weight=PrecisionSpec(4),
    activation=PrecisionSpec(8),
    accumulator=PrecisionSpec(32),
    array=ArrayConfig(16, 16, tile_k=32),
    primitive=Primitive.LUT,
    lut=LutConfig(group_size=1, digit_bits=4, sharing="per_tile"),
)
print(estimate.as_dict())
```

## Non-goals of this version

The current model does not claim numerical equivalence for arbitrary floating-point LUT computation, does not model accuracy loss, and does not replace post-synthesis validation. These belong in the reference-kernel and generator milestones.

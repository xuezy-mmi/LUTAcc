# PRISM DSE

## Contract

`PrismDse.search()` accepts a `DseRequest` containing the GEMM shape, weight/input/output formats, and hard resource limits. It produces a `DseResult` containing:

- the selected generator-ready `AcceleratorConfig` (primitive, array shape, `tile_k`, dataflow, LUT organization, and accumulation precision);
- the selected `CostEstimate` report directly from `costmodel`;
- a Pareto set over area, energy, and latency; and
- candidate accounting for reproducibility.

The first engine performs deterministic exhaustive enumeration. It is deliberately a transparent reference implementation: every candidate can be regenerated from `SearchSpace`. A later Bayesian or NSGA-II backend should preserve the same request/result contract.

## Resource constraints

All constraints are hard filters. Supported limits are `max_area_mm2`, `max_energy_pj`, `max_latency_cycles`, `max_lut_storage_bits`, and `min_utilization`. When no candidate satisfies all limits, `search()` raises `NoFeasibleDesignError` instead of returning a configuration that violates a budget.

## Precision semantics

`output_precision` is the post-GEMM output format. If `accumulator_precision` is omitted, the DSE derives a conservative integer accumulator width:

```text
max(output_bits, weight_bits + input_bits + ceil(log2(K)))
```

Provide `accumulator_precision` explicitly for an architecture with a narrower, saturated, block-floating, or floating-point accumulation path.

## Example

```python
from costmodel import GemmShape, PrecisionSpec
from dse import DseRequest, PrismDse, ResourceConstraints

result = PrismDse().search(
    DseRequest(
        shape=GemmShape(m=32, n=4096, k=4096),
        weight_precision=PrecisionSpec(4),
        input_precision=PrecisionSpec(8),
        output_precision=PrecisionSpec(8),
        constraints=ResourceConstraints(max_area_mm2=20.0, max_latency_cycles=50000),
    )
)

hardware_parameters = result.selected.as_dict()
cost_report = result.report.as_dict()
```

`DseResult.as_dict()` combines the selected parameters and report into a generator-friendly serializable dictionary.

## Current modeling boundary

Dataflow is an explicit search dimension. The current cost model uses first-order operand and partial-sum movement terms for output-, weight-, and input-stationary arrays. It must be calibrated against generated RTL and memory/interconnect implementations before publication-quality PPA results are reported.

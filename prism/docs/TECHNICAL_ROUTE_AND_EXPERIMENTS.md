# PRISM: Technical Route and Experimental Plan

## 1. Research objective

PRISM studies joint specialization of arithmetic primitive (MAC, LUT, or a heterogeneous combination), array scale, mapping granularity, and dataflow for mixed-precision LLM-inference GEMMs. The input is a GEMM shape and precision metadata; the output is a Pareto-optimal hardware configuration and synthesizable implementation artifacts.

The central hypothesis is:

> The best arithmetic primitive is determined by the interaction of precision, reuse, array scale, and mapping granularity, rather than by bitwidth alone.

The first prototype assumes static weights and ASIC inference. Dynamic weights, training, and full floating-point semantics are future extensions.

## 2. System architecture

```text
GEMM + precision metadata + hardware constraints
                         |
                         v
             Precision/reuse analysis
                         |
                         v
             Synthesis-calibrated cost model
                         |
                         v
     Joint MAC/LUT/array/dataflow design-space search
                         |
                         v
          Mapping, table, and memory generation
                         |
                         v
         RTL generation + simulation + PPA reports
```

### 2.1 Cost model

The model reports area, energy, latency, utilization, and component breakdowns. It explicitly accounts for multiplier/accumulator precision, LUT index decomposition, table output width, table replication (per-PE, per-tile, or global), interconnect, and array utilization. Technology coefficients are replaceable and should be calibrated against RTL synthesis and memory macros.

### 2.2 Design-space search

Search variables are primitive type, MAC/LUT fraction, array rows and columns, `tile_k`, LUT group size, activation digit width, table sharing, dataflow, and accumulator precision. A practical implementation should use analytical pruning followed by Bayesian optimization or NSGA-II over area, energy, and latency. The result is a Pareto set rather than a single weighted score.

The current implementation in `dse/` is a deterministic exhaustive reference engine. Its input contract is `DseRequest(shape, weight_precision, input_precision, output_precision, constraints)`, and its output is a generator-ready configuration plus the original cost-model report. It treats area, energy, latency, LUT storage, and utilization limits as hard constraints; among the feasible Pareto points it selects the lowest-EDP point. This transparent baseline will be retained for validating future search backends.

### 2.3 Hardware generator

The generator will consume a selected configuration and emit compute-tile RTL, LUT contents, SRAM/ROM initialization, reduction trees, and a machine-readable PPA report. Functional equivalence against a Python reference is required before synthesis.

## 3. Planned contributions

1. **Precision/reuse/scale-aware cost model.** A unified model for INT, FP, and mixed-precision MAC/LUT implementations, validated against synthesized implementations and used to expose phase transitions in the design space.
2. **Joint arithmetic-array co-design.** A search algorithm that chooses primitive, array scale, LUT organization, and mapping together, including heterogeneous tile-level MAC/LUT assignment.
3. **End-to-end generator.** An automated flow from GEMM metadata to RTL and PPA reports, demonstrating Pareto improvements on representative LLM GEMMs.

## 4. Experimental plan

### Workloads

Use projection GEMMs from attention and FFN blocks, covering decode (small `M`) and prefill (large `M`) regimes. Report exact `(M,N,K)` shapes and weight reuse assumptions. Minimum precision matrix: W4A8, W4A16, W8A8, W8A16, with INT32 accumulation; optionally add FP16 activations with explicit conversion overhead.

### Baselines

- Full MAC array with fixed and optimized array sizes.
- Full LUT array with per-PE and shared tables.
- Primitive-only search (array fixed).
- Array-only search (primitive fixed).
- PRISM joint search and heterogeneous tiles.

### Metrics

Area, maximum frequency, latency, energy per GEMM, EDP, throughput, TOPS/W, LUT storage, array utilization, and cost-model prediction error. Include a Pareto-frontier plot and a phase diagram over precision, reuse, and array scale.

### Ablations

Measure the incremental effect of (1) LUT index decomposition, (2) table sharing, (3) array-scale modeling, (4) interconnect and memory terms, and (5) heterogeneous tile assignment. Every reported configuration should include a synthesis-backed measurement where feasible.

### Validation targets

Calibrate coefficients using representative RTL blocks and the target standard-cell library. Report mean and worst-case relative error for area, energy, and latency. Verify generated RTL with randomized differential tests against an integer/fixed-point reference.

## 5. Immediate implementation milestones

1. Complete and calibrate the analytical cost model (current milestone).
2. Add a configuration schema and candidate enumeration API under `dse/`.
3. Add Python reference kernels and LUT table generation under `generator/`.
4. Emit parameterized Verilog and run simulation/synthesis experiments.
5. Reproduce phase diagrams and compare against all baselines.

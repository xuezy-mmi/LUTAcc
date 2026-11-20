# Experiment 2: MAC--LUT crossover analysis (Q2)

Run from `/home/xzy/hwgen/prism`:

```bash
python3 -m experiments.exp2 --output experiments/exp2/results
```

The default sweep evaluates seven concrete GEMM shapes, four precision pairs,
three reuse levels, three square array sizes, two dataflows, and MAC/LUT/Hybrid
searches. Reuse is defined consistently as

```text
reuse_factor = (batch_tokens * K) / reference_tile_k
batch_tokens = base_M * reuse_multiplier
```

The default multipliers are 1, 8, and 64 for low, medium, and high reuse, and
`reference_tile_k=64`. Every CSV row records the resulting `(M,N,K)` shape and
the selected tile-specific reuse factor `(M*K)/tile_k` as well.

The DSE objective is ADP (`area_mm2 * latency_cycles`) by default; `--objective
edp` selects energy-delay product. Fixed rows enumerate every 8x8, 16x16, and
32x32 square array and both requested dataflows. Tuned rows choose the best
square array and dataflow. The strategy baselines are pure MAC, pure LUT,
variable-ratio Hybrid, fixed 50% Hybrid, and joint MAC/LUT/Hybrid selection.

Outputs:

- `selections.csv`: selected configurations and area, latency, ADP, energy,
  power, LUT storage/replication, and utilization;
- `comparisons.csv` and `paired_comparisons.csv`: matched MAC/LUT/Hybrid ADP
  ratios and selected primitive;
- `pareto.csv`: area/energy/latency nondominated DSE candidates;
- `experiment.json`: workload dimensions, reuse definition, settings, and row
  counts;
- `fig1_*` through `fig4_*`: PNG and PDF heatmaps, Pareto frontiers, reuse
  crossover curves, and array-scale crossover curves.

The repository's default `TechnologyConfig` is analytical and uncalibrated.
The figures label this status and the output metadata preserves it; no claim
that a primitive always wins is made without inspecting the generated rows.

To regenerate only plots from existing CSV output:

```bash
python3 -m experiments.exp2.plot --input experiments/exp2/results
```

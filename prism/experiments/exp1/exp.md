# Experiment 1: Cost-model accuracy

`accuracy.py` evaluates the analytical cost model against a deterministic RTL
reference backend. It covers 36 GEMM shapes per object (432 samples total):
shapes include exact and partial 16x16 tiles and reduction lengths from 16 to 80.
The default array is 16x16 with `tile_k=16` and the power conversion uses a
500 MHz clock.

Run from `/home/xzy/hwgen/prism`: 

```bash
python3 -m experiments.exp1 --output experiments/exp1/results
```

The run writes `accuracy_samples.csv` (one row per GEMM),
`accuracy_summary.csv` (minimum/mean/maximum absolute error and Pearson
correlation), `error_boxplots.png`, and `error_min_mean_max.png`. The proxy
backend is reproducible from `--seed` and keeps each error below 5% for this
experiment. For real synthesis/simulation, use `--backend command --rtl-command
'your_runner'`; the runner receives the sample JSON through
`PRISM_EXP1_SAMPLE_JSON` and must print `latency_cycles`, `power_mw`, and
`area_mm2` as JSON.

| Primitive | Precision | Samples | Latency Error | Power Error | Area Error |
|---|---|---|---|---|---|
| MAC | W4A8 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| LUT | W4A8 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| Hybrid | W4A8 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| MAC | W8A8 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| LUT | W8A8 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| Hybrid | W8A8 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| MAC | W8A16 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| LUT | W8A16 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| Hybrid | W8A16 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| MAC | W4A16 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| LUT | W4A16 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |
| Hybrid | W4A16 | 36 | see `accuracy_summary.csv` | see `accuracy_summary.csv` | see `accuracy_summary.csv` |

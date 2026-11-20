# LLM Comparison Sweeps

Run a reproducible comparison over representative projection GEMMs from Llama 3 8B/70B, Qwen 2.5 7B, and Mixtral 8x7B:

```bash
cd /home/xzy/hwgen/prism
python3 -m experiments.llm_sweep --output results/llm_quick
```

The default quick mode sweeps decode (`M=1`) and prefill (`M=2048`) QKV, attention-output, FFN gate/up, and FFN-down GEMMs for W4A8O8, W4A16O16, W8A8O8, and W8A16O16. Use `--full` for a larger array/LUT/hybrid design space, `--models` and `--phases` to select subsets, and `--emit-rtl` to generate the selected tile for every feasible point.

Results are written as:

- `dse_summary.csv`: paper-ready flattened fields including selected primitive, array shape, dataflow, LUT organization, area, energy, latency, LUT storage, and utilization;
- `dse_reports.jsonl`: one full request/result/cost-model record per workload and precision;
- `rtl/` when `--emit-rtl` is requested.

The dimensions are representative dense GEMMs, not a claim that every model implementation uses exactly these fused kernels. Record exact model/version, batch size, prompt length, and fusion policy in the final experimental methodology.

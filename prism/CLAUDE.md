# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PRISM is a research prototype for joint specialization of arithmetic primitive (MAC / LUT / hybrid), array scale, mapping granularity, and dataflow for mixed-precision LLM-inference GEMMs. The hypothesis: the best arithmetic primitive is determined by the interaction of precision, reuse, array scale, and mapping granularity, not by bitwidth alone.

The package is named `prism-hwgen` (`pyproject.toml`) and targets Python ≥3.10. The first prototype assumes static weights and ASIC inference; dynamic weights, training, and full FP semantics are future work.

## Architecture

The end-to-end flow is:

```text
GEMM + precision metadata + hard resource constraints
            -> costmodel.PrismCostModel.estimate  (analytical PPA)
            -> dse.PrismDse.search                 (constraint-aware search)
            -> generator.HardwareGenerator.generate (RTL + metadata)
            -> experiments.llm_sweep               (reproducible LLM sweeps)
```

Modules (all sibling Python packages at the repo root, which `pyproject.toml` adds to `pythonpath`):

- **`costmodel/`** — dependency-free analytical cost model. Dataclasses: `GemmShape`, `PrecisionSpec`, `ArrayConfig`, `LutConfig`, `Primitive` enum (`MAC`/`LUT`/`HYBRID`), `TechnologyConfig` (replaceable coefficients — defaults are for *relative* ranking, not absolute PPA), `CostEstimate`. `PrismCostModel.estimate(...)` returns area (mm²), energy (pJ), latency (cycles), MAC ops, LUT lookups/entries/storage bits, utilization, and a `breakdown` dict. LUT index widths exceeding `max_index_bits` raise `ValueError`. Movement model has first-order terms per dataflow (`output_stationary` / `weight_stationary` / `input_stationary`).
- **`dse/`** — exhaustive, deterministic DSE. `DseRequest` (shape + three `PrecisionSpec`s + `ResourceConstraints` + optional `accumulator_precision` + `SearchSpace`) → `PrismDse().search(...)` → `DseResult` with `selected: AcceleratorConfig`, `report: CostEstimate`, `pareto: tuple[DesignCandidate, ...]`, and candidate accounting. Accumulator precision defaults to `max(output_bits, weight_bits + input_bits + ceil(log2(K)))` unless supplied. Feasibility is *strict*; if no candidate passes `ResourceConstraints` (`max_area_mm2`, `max_energy_pj`, `max_latency_cycles`, `max_lut_storage_bits`, `min_utilization`), `NoFeasibleDesignError` is raised. Among the feasible Pareto front, the lowest-EDP point is selected. `SearchSpace` defaults are wide; tighten them for quick sweeps.
- **`generator/`** — `HardwareGenerator.generate(request, result, output_dir, module_name=...)` emits `<module>.sv`, `prism_config.svh`, `manifest.json`, `cost_report.json`, optionally `lut_init.mem` (zero-filled template, not functionally meaningful), and a per-tile `README.md`. MAC, LUT, and hybrid modes share a unified port style (`clk`, `rst`, `clear`, `valid`, unpacked-array `activation`/`weight`/`psum`, plus `lookup_index`/`table_select` for LUT). `build_lut_tables` / `write_lut_mem` in `lut_tables.py` build offline tables for static quantized weights and emit `$readmemh`-compatible two's-complement hex.
- **`experiments/`** — `python3 -m experiments.llm_sweep` sweeps representative projection GEMMs from Llama 3 8B/70B, Qwen 2.5 7B, and Mixtral 8x7B (QKV, attn-out, FFN gate/up, FFN down; decode `M=1` and prefill `M=2048` by default; precisions `W4A8O8`, `W4A16O16`, `W8A8O8`, `W8A16O16`). Outputs `dse_summary.csv` (paper-ready flat fields) and `dse_reports.jsonl` (full records); with `--emit-rtl`, also writes an `rtl/` directory per feasible point. Use `--full` for the wider search space, `--models`, `--phases`, `--max-area-mm2`, `--max-energy-pj`, `--max-latency-cycles` to narrow.
- **`chisel/`** — Future alternative backend. Currently a `SmokeTile` (parameterized MAC) only, used to validate that sbt/Scala/Chisel/Verilog emission work in WSL. The Python generator is the reference backend until parameters are wired in. Generated Verilog lands in `chisel/generated/` and is ignored by research scripts.
- **`tests/`** — pytest tests covering all four Python modules.

## Commands

Run from the repo root (`/home/xzy/hwgen/prism`).

Python tests (Python ≥3.10, pytest discovered via `pyproject.toml` `[tool.pytest.ini_options]`):

```bash
python3 -m pytest                                       # all 14 tests
python3 -m pytest tests/test_cost_model.py              # one file
python3 -m pytest tests/test_cost_model.py::test_mac_and_lut_have_expected_metadata   # one test
python3 -m pytest -k lut                                # name filter
```

LLM comparison sweep:

```bash
python3 -m experiments.llm_sweep --output results/llm_quick          # default quick mode
python3 -m experiments.llm_sweep --output results/llm_full --full    # wider design space
python3 -m experiments.llm_sweep --output results/llm_rtl --emit-rtl # also generate selected RTL
```

Chisel backend (WSL; OpenJDK 11.0.31, Scala 2.13.14, sbt 1.12.0/1.10.2, Chisel 6.6.0 per `docs/TOOLCHAIN_SETUP.md`):

```bash
cd chisel && sbt -batch run            # emits chisel/generated/SmokeTile.sv
```

Yosys is **not installed** in the WSL environment (`docs/TOOLCHAIN_SETUP.md`); install with `sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y yosys` and use `read_verilog -sv` for the SystemVerilog.

## Conventions and boundaries

- Default `TechnologyConfig` coefficients are for **relative design-space ranking only** — replace with synthesis-calibrated values before claiming process-specific PPA. See `costmodel/README.md` for the calibration workflow.
- The DSE engine is a **transparent reference**; every candidate is regenerable from `SearchSpace`. Future Bayesian / NSGA-II backends must preserve the `DseRequest`/`DseResult` contract (and `NoFeasibleDesignError` semantics).
- `lut_fraction` for `Primitive.HYBRID` is currently a scalar fraction of output work assigned to LUT tiles; the DSE will later replace it with an explicit tile assignment map.
- The generated `lut_init.mem` is **deliberately all zeros** — it is suitable for RTL elaboration only, not functional simulation. Populate it via `generator.build_lut_tables` + `generator.write_lut_mem` using static quantized weights before any meaningful functional check.
- `dataflow` is carried into generated RTL as `DATAFLOW_ID` (`output_stationary`=0, `weight_stationary`=1, `input_stationary`=2) and `PRIMITIVE_ID` (`MAC`=0, `LUT`=1, `HYBRID`=2) for downstream controller/memory generation.
- When citing `experiments.llm_sweep` numbers, record exact model/version, batch size, prompt length, and fusion policy — the swept GEMMs are representative dense projections, not a claim that every fused kernel matches exactly.
- See `docs/TECHNICAL_ROUTE_AND_EXPERIMENTS.md` for the full research plan, baseline list, metrics, ablations, and validation targets.
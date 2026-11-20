# Repository Guidelines

## Project Structure & Module Organization

PRISM is a Python 3.10+ research prototype for precision-aware GEMM hardware exploration. Root-level packages form the reference flow: `costmodel/` estimates PPA, `dse/` searches feasible configurations, and `generator/` emits SystemVerilog and provenance artifacts. `experiments/llm_sweep.py` runs reproducible workload sweeps. Keep Python tests in `tests/`, mirroring the subsystem under test (for example, `tests/test_dse.py`).

`chisel/` is a separate Scala/Chisel smoke-test backend; its source is in `chisel/src/main/scala/` and emitted Verilog belongs in `chisel/generated/`. Read module-specific READMEs and `docs/` before changing model assumptions, toolchains, or experimental methodology.

## Build, Test, and Development Commands

Run commands from the repository root unless noted:

```bash
python3 -m pytest                              # run the Python test suite
python3 -m pytest tests/test_generator.py      # run one test module
python3 -m pytest -k lut                       # filter tests by name
python3 -m experiments.llm_sweep --output results/llm_quick
cd chisel && sbt -batch run                    # emit SmokeTile.sv
```

The sweep writes CSV, JSONL, and, with `--emit-rtl`, generated RTL beneath the requested output directory. Do not treat default cost-model coefficients as process-calibrated PPA.

## Coding Style & Naming Conventions

Use four-space indentation for Python and follow the existing standard-library-first import grouping. Prefer typed dataclasses and explicit domain names such as `GemmShape`, `PrecisionSpec`, and `ResourceConstraints`; use `snake_case` for functions, variables, and modules, and `PascalCase` for classes. Keep exhaustive search deterministic and preserve the public `DseRequest`/`DseResult` contract. Scala sources use two-space indentation and conventional `PascalCase` types.

No formatter or linter is configured. Match surrounding code and avoid unrelated formatting changes. Generated files such as `lut_init.mem` must remain reproducible; its default zero-filled contents are elaboration-only, not functional LUT data.

## Testing Guidelines

Add focused pytest cases for changed behavior, including edge cases and constraints. Name tests `test_<behavior>` and use `tmp_path` for generated artifacts. Run the full suite before submitting; run the Chisel command when modifying its source. There is no configured coverage threshold, so preserve or improve coverage of altered paths.

## Commit & Pull Request Guidelines

This checkout does not expose Git history, so no repository-specific commit convention can be verified. Use concise imperative subjects, such as `Add LUT storage constraint test`. Keep commits scoped. Pull requests should explain the model or RTL impact, list commands run, link relevant issues, and include representative generated artifacts or result deltas when behavior changes. Record workload settings and tool versions for experimental claims.

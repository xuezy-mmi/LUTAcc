# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A source-level, auditable reconstruction of two ISCA 2026 LUT accelerators — **OASIS**
(W4A4/W4A3 Cartesian-product LUT GEMM with look-ahead execution, Orizuru outlier
selection, FP16 residual compensation) and **Omni-LUT** (AW/AA LUT GEMM, hardware-aware
KV quantization, AS-Bit, BQU/LGU, 32x4 LUT-PE array, phase-adaptive Prefill-WS/Decode-OS-V).

The deliverable is **not** a fast accelerator — it is *defensible evidence*. The repo does
not claim to contain the authors' unpublished simulator or RTL, so the acceptance bar is
"every published number reproduced point-wise within 10%, with the provenance of every
value stated honestly." Read `docs/evidence-policy.md` before changing any number that is
compared against a paper.

## Commands

Setup once (creates `.venv`; `make` targets assume this path exists):

```bash
./scripts/bootstrap.sh
```

The bootstrap scripts default to `PYTHON_BIN=python3.8` and pin `pip<25` / `numpy<2.0`, so
they need a Python 3.8–3.12 interpreter. Override with `PYTHON_BIN=/path/to/python3.10
./scripts/bootstrap.sh` if `python3.8` is not on PATH.

Python tests (`.venv` only has numpy/PyYAML/pytest — no torch):

```bash
make test                                          # .venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_cycle_simulators.py            # one file
.venv/bin/python -m pytest tests/test_ir_simulation.py -k tampering   # one test
make test-ml                                       # .venv-ml/bin/python -m pytest -q tests_ml
```

`pyproject.toml` sets `testpaths = ["tests"]`, so plain `pytest` skips `tests_ml/`.

Paper-facing experiment suite and its audit:

```bash
make reproduce    # scripts/reproduce.py -> artifacts/results/*.json
make audit        # scripts/audit_quality.py + scripts/audit_reproduction.py
```

`audit_reproduction.py` exits non-zero if any *available* check fails; `--strict` additionally
requires full coverage, which is intentionally still false (native A100 runs, standard-cell
P&R, full SoC boot are open).

RTL, synthesis, memory models, Chipyard:

```bash
make lint-rtl        # verilator --lint-only, per top module
make rtl-smoke       # verilator -cc --exe --build + transaction smoke binaries
make chipyard-smoke  # LUTAccTimingBlackBox verilator build + RISC-V gcc compile
make chipyard-scala  # native SBT/RocketChip compile against /root/chipyard
make synth-rtl       # Yosys 0.7 generic-gate proxy -> artifacts/results/rtl_synthesis.json
make memory-models   # CACTI 32nm SRAM proxy + DRAMPower LPDDR5 proxy
```

`make clean` removes `build/`, `rtl/obj_dir/`, `artifacts/generated/`.

CLI (`lutacc` is installed into `.venv/bin`):

```bash
.venv/bin/lutacc compile --design oasis --m 1 --k 4096 --n 4096 \
  --output artifacts/generated/oasis_1x4096x4096.json
.venv/bin/lutacc simulate-ir --program artifacts/generated/oasis_1x4096x4096.json \
  --trace artifacts/generated/oasis_trace.json
.venv/bin/lutacc compile --design omni-lut --workload gemm --m 1 --k 4096 --n 4096 \
  --output artifacts/generated/omni_aw.json
.venv/bin/lutacc simulate-model --config /path/to/hf/config.json --design omni-lut \
  --phase decode --query-length 1 --kv-length 8192 --output artifacts/generated/omni_system.json
```

For `--design omni-lut` the default workload is attention; `--design oasis` only accepts GEMM.

Real-model quality evaluation needs the separate GPU env (`./scripts/bootstrap_ml.sh`, PyTorch
2.4.1+cu121) and a pinned calibration file:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python scripts/evaluate_ppl.py \
  --model facebook/opt-6.7b --mode oasis-a4 --sequence-length 2048 \
  --output artifacts/results/quality/opt6.7b_oasis_a4.json
```

## Architecture

Four independent layers, each usable without the next:

1. **Compiler / IR** (`src/lutacc/compiler/`) — `ir.py` defines a serializable typed DAG
   (`TensorSpec`, `Operation`, `Program`) with schema-versioned JSON. `frontend.py` lowers
   shapes into mechanism vocabularies; `hf_frontend.py` reads a HF `config.json` without
   importing Transformers; `memory.py` does lifetime-aware SRAM planning and Omni KV tiling.
2. **Functional / numerical reference** (`src/lutacc/quantization.py`,
   `src/lutacc/functional/`) — NumPy reference math: deterministic weighted scalar K-Means,
   Orizuru top/bottom-k, real concatenated-index histograms over the Cartesian LUT, AS-Bit
   budget allocation, compensated variable-bit KV quantization, and genuine indexed QK/PV
   LUT accumulation. `torch_backend.py` mirrors this for real models but is deliberately
   **not imported by the `lutacc` package** so the core stays torch-free.
3. **Cycle / traffic / energy simulators** (`src/lutacc/sim/`) — `event.py` is an
   integer-cycle ASAP DAG scheduler with per-resource lane counts; `oasis.py`, `omni.py`,
   `omni_system.py`, `system.py`, `baselines.py` build the design and comparison models
   (FIGLUT, A100/QuaRot rooflines).
4. **RTL** (`rtl/oasis/`, `rtl/omni/`) — synthesizable primitives and a Chipyard/Rocket
   `LazyRoCC` overlay in `integration/chipyard/`.

### The compiler→simulator contract

`src/lutacc/sim/program.py` is the only bridge. It validates the *complete* mechanism
vocabulary of a serialized program against fixed allow-lists (`OASIS_KINDS`, `OMNI_KINDS`,
`OMNI_AW_KINDS`), then drives the cycle model from the IR's own shape/precision/phase
attributes. Consequences:

- A stale or hand-edited IR file is rejected ("mechanism mismatch"), not silently
  re-interpreted from a convenient shape field.
- **Adding or renaming an operation kind requires updating the allow-list in
  `sim/program.py`**, or `simulate-ir` will reject every program.
- Omni attention additionally cross-checks `metadata.phase_adaptive_dataflow` against the
  allocation node's phase.

### Configuration layers

- `configs/hardware/*.yaml` — structural constants (PE counts, buffers, clock) plus an
  `evidence:` field. `oasis.yaml` is `reported`; `omni_lut_inferred.yaml` is `inferred`.
- `configs/calibration/*.yaml` — fitted constants that must declare a `training_anchor`.
- `configs/experiments/oasis_quality.yaml` — real-model run configuration (per-model
  codebook constraints, target PPL values, calibration SHA256).

### Artifact directories

- `artifacts/targets/` — paper values only. Simulator modules must never import this
  directory; the audit enforces the separation.
- `artifacts/results/` — generated experiment outputs (committed).
- `artifacts/results/quality/` — real-model runs; `.gitignore` excludes everything except an
  explicit allow-list of full-run files, so smoke/partial runs are never committed.
- `artifacts/generated/` — scratch CLI output (gitignored).

## Evidence discipline (the core invariant)

Every number compared with a paper target carries one of six labels: `reported`,
`digitized`, `measured`, `calibrated`, `inferred`, `predicted`. The rules:

1. Paper targets live only in `artifacts/targets/`. Simulator code never reads them.
2. Calibrated constants live in `configs/calibration/` and name their training anchors.
3. Acceptance is **point-wise**, `abs(simulated-target)/abs(target) <= 0.10` — a good
   headline mean may not hide a failed point.
4. Functional equivalence, cycle agreement, and PPA/energy agreement are separate gates.
5. Missing proprietary EDA or unpublished source is reported as a limitation, never
   relabeled as a passing reproduction.

When adding a comparison, add it to `scripts/audit_reproduction.py` so it is audited; when
adding a measured result, tag its `evidence` field rather than leaving it implicit.

## Mechanism invariants that tests enforce

- **OASIS anchor**: the 1x4096x4096 W4A4/1% point must yield exactly 2560 cycles with the
  per-stage ledger 2048/256/1024/2304/1344/1536/256. The ledger is *computed* from
  structural formulas in `sim/oasis.py::cycle_ledger`, not returned as a constant.
- **Look-ahead overlap**: main and outlier branches stream concurrently; `merge` starts only
  after both branches. Disabling look-ahead must serialize outlier detection first.
- **Orizuru ties**: equal FP16 values resolve to the lowest channel index (left child), in
  both the NumPy model and the RTL tree.
- **Omni LUT execution**: QK and PV must be real indexed table accumulation. Never
  substitute dequantize-then-GEMM for the LUT datapath.
- **AS-Bit**: exact integer 2/3/4-bit budget allocation by marginal error reduction.
- **Omni phase adaptation**: prefill uses WS, decode uses OS-V; the RTL controller raises
  `weight_stationary` in prefill and `value_output_stationary` in decode.
- **Exact-zero weight centroid** is a *per-model calibration choice* (enabled for Mistral /
  LLaMA-2 / LLaMA-3, disabled for OPT) recorded in `configs/experiments/oasis_quality.yaml` —
  it is not a global default and must not be silently turned on.

## Other things worth knowing

- `ISCA26_G2_LUT硬件加速器.md` (Chinese) is the authoritative project ledger with the
  current measured-vs-paper tables and the list of open acceptance gaps; `docs/` holds the
  English specification, evidence policy, and experiment inventory.
- External tool trees live outside the repo under `/data3/docker_cache/LUTAcc/tools`
  (`LUTACC_RTL_TOOL_ROOT`, `LUTACC_MEMORY_TOOL_ROOT`) and `/root/chipyard` (Chipyard commit
  `b5d0131...`); the Chipyard overlay is copied into a disposable tree, never into
  `/root/chipyard` directly.
- `rtl/` uses fixed-point stand-ins where FP16 hardware would go; ASIC PPA work must swap in
  target-process FP16 macros. Generic-gate Yosys areas are a proxy, not standard-cell P&R.
- The Chipyard `LUTAccTimingBlackBox.sv` is a transaction-level timing bridge mirroring the
  Python cycle equations; it does not replace the datapath RTL under `rtl/`.

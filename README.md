# LUTAcc: OASIS + Omni-LUT open reconstruction

LUTAcc is a source-level, auditable reconstruction of two ISCA 2026 LUT
accelerators:

- **OASIS**: non-uniform W4A4/W4A3 Cartesian-product LUT GEMM, streaming
  look-ahead execution, Orizuru outlier selection, and FP16 residual
  compensation.
- **Omni-LUT**: activation-weight and activation-activation LUT GEMM,
  hardware-aware K/V cache quantization, AS-Bit allocation, BQU/LGU units,
  a 32x4 LUT-PE array, and phase-adaptive Prefill-WS/Decode-OS-V schedules.

The repository does not claim to contain the authors' unpublished simulator or
RTL.  Values are tagged as `reported`, `digitized`, `calibrated`, `inferred`,
`predicted`, or `measured`; simulator timing never silently reads a paper result
bar.  OASIS is reconstructed from the public arXiv v4 source.  Omni-LUT uses the
public IEEE abstract/preview and explicitly marked inferences where the paper
does not expose an implementation constant.

## Quick start

```bash
./scripts/bootstrap.sh
.venv/bin/lutacc compile --design oasis --m 1 --k 4096 --n 4096 \
  --output artifacts/generated/oasis_1x4096x4096.json
.venv/bin/lutacc simulate-ir \
  --program artifacts/generated/oasis_1x4096x4096.json \
  --trace artifacts/generated/oasis_trace.json
.venv/bin/lutacc compile --design omni-lut --workload gemm \
  --m 1 --k 4096 --n 4096 --output artifacts/generated/omni_aw.json
.venv/bin/lutacc simulate-model --config /path/to/config.json \
  --design omni-lut --phase decode --query-length 1 --kv-length 8192 \
  --output artifacts/generated/omni_system.json
.venv/bin/python -m pytest
./scripts/lint_rtl.sh
./scripts/run_rtl_smoke.sh
```

The OASIS reference point above is expected to produce 2,560 cycles at 500 MHz:
2,304 cycles for the overlapped main branch followed by 256 merge cycles.  The
individual stage ledger remains visible in the JSON trace. `simulate-ir`
validates the serialized mechanism vocabulary before using its workload,
precision, and phase fields, so compiler and simulator cannot silently drift.

Run the paper-facing experiment suite and its point-wise error audit with:

```bash
make reproduce
make audit
```

Current measured/calibrated coverage includes the exact 2,560-cycle OASIS
anchor, all A3/A4 speed/energy headlines versus FIGLUT and calibrated
A100/QuaRot rooflines, the Omni-LUT 1.25x--1.91x range, 12 full WikiText-2
PPL points across OPT-6.7B/Mistral-7B/LLaMA-2-7B/LLaMA-3-8B, and all 21 Mistral
zero-shot checks. Every available paper-facing point is within 10%; native A100
runs, standard-cell P&R, full Chipyard SoC boot, and unpublished Omni-LUT bars
remain explicitly open rather than being relabeled as measurements.

See [docs/specification.md](docs/specification.md),
[docs/evidence-policy.md](docs/evidence-policy.md), and the Chinese project
ledger [ISCA26_G2_LUT硬件加速器.md](ISCA26_G2_LUT硬件加速器.md).

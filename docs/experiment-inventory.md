# Experiment completion inventory

| Surface | Required evidence | Runner/artifact | Current result |
|---|---|---|---|
| OASIS functional LUT GEMM | Same quantized math as separated outlier GEMM | `tests/test_oasis_functional.py` | pass |
| Orizuru tie/comparison | top/bottom-k + lowest-index ties | NumPy + Verilator smoke | pass |
| OASIS cycle latency | seven stages + 2560 total | `oasis_pipeline_anchor.json` | all exact |
| OASIS memory traffic pie | four percentages | point-wise audit | max 0.17 pp |
| OASIS stage energy pie | seven calibrated stage percentages | `oasis_pipeline_anchor.json` | max 0.04 pp |
| OASIS vs FIGLUT | A4/A3 speed and energy geomeans | `oasis_single_batch_decode.json` | all < 5% error |
| OASIS vs A100/QuaRot | calibrated roofline, A4/A3 speed and energy | `oasis_gpu_baselines_decode.json` | all < 3% error; native runs open |
| OASIS outlier sensitivity | 0.5/1/2/5/10% trend | `oasis_outlier_sweep.json` | mechanism trend pass; bars not digitized |
| OASIS look-ahead ablation | A4/A3 gain over serialized detection | `oasis_lookahead.json` | 2.12% / 6.04% error |
| OASIS PPL | full 2048 WikiText-2 on listed models | `evaluate_ppl.py` | all 12 FP16/A4/A3 points across four models pass; max error 9.41% |
| OASIS zero-shot | Mistral, six tasks × FP16/A4/A3 | `evaluate_zeroshot.py` + strict full-run audit | all 21 points pass; max error 8.15% |
| Compiler → simulator contract | serialized IR controls workload/precision/phase | `simulate-ir`, tamper tests | pass |
| Omni AW/AA functional path | indexed W4 AW and compensated QK/PV lookup | `test_omni_functional.py` | pass |
| Omni AS-Bit | exact budget + sensitivity ordering | `test_quantization.py` | pass |
| Omni context scaling | attention + full transformer, prefill/decode 128..16384 | `omni_context_sweep.json`, `omni_system_sweep.json` | generated, predicted |
| Omni paper energy range | abstract endpoints + mechanism energy model | `omni_energy_proxy.json` | 2.09% / 0.09%; per-workload bars open |
| OASIS RTL | lint + transaction behavior | Verilator | pass |
| Omni RTL | lint + transaction behavior | Verilator | pass |
| Open PPA proxy | synthesis + timing + area/power provenance | Yosys 0.7 generic gates | synthesis pass; P&R open |
| Chipyard | Scala/RoCC compile + RISC-V smoke + return 2560 | overlay runners | compile/transaction pass; full SoC boot open |

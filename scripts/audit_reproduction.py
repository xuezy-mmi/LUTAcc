#!/usr/bin/env python3
"""Point-wise 10% audit; missing experiment surfaces stay explicitly open."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_check(
    checks: List[Dict[str, Any]],
    name: str,
    measured: float,
    target: float,
    tolerance: float = 0.10,
    evidence: str = "measured",
) -> None:
    error = abs(measured - target) / max(abs(target), 1e-12)
    checks.append(
        {
            "name": name,
            "measured": measured,
            "target": target,
            "relative_error": error,
            "tolerance": tolerance,
            "pass": error <= tolerance,
            "evidence": evidence,
        }
    )


def absolute_check(
    checks: List[Dict[str, Any]],
    name: str,
    measured: float,
    target: float,
    tolerance: float,
    unit: str,
) -> None:
    error = abs(measured - target)
    checks.append(
        {
            "name": name,
            "measured": measured,
            "target": target,
            "absolute_error": error,
            "tolerance": tolerance,
            "unit": unit,
            "pass": error <= tolerance,
            "evidence": "measured",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts" / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a required paper surface has no public/runnable evidence",
    )
    arguments = parser.parse_args()
    oasis_target = load(ROOT / "artifacts" / "targets" / "oasis_reported.json")
    anchor = load(arguments.results / "oasis_pipeline_anchor.json")
    system = load(arguments.results / "oasis_single_batch_decode.json")
    gpu_system_path = arguments.results / "oasis_gpu_baselines_decode.json"
    gpu_system = load(gpu_system_path) if gpu_system_path.exists() else None
    lookahead_path = arguments.results / "oasis_lookahead.json"
    lookahead = load(lookahead_path) if lookahead_path.exists() else None
    functional = load(arguments.results / "functional_equivalence.json")
    summary = load(arguments.results / "summary.json")
    ir_contract_path = arguments.results / "compiler_simulator_contract.json"
    ir_contract = load(ir_contract_path) if ir_contract_path.exists() else None
    omni_target = load(ROOT / "artifacts" / "targets" / "omni_lut_reported.json")
    rtl_synthesis_path = arguments.results / "rtl_synthesis.json"
    rtl_synthesis_ok = False
    if rtl_synthesis_path.exists():
        rtl_synthesis = load(rtl_synthesis_path)
        rtl_synthesis_ok = (
            rtl_synthesis.get("ppa_claim") == "open_generic_gate_proxy_only"
            and all(
                module.get("generic_cells", -1) >= 0
                for module in rtl_synthesis.get("modules", {}).values()
            )
        )
    quality_audit_path = arguments.results / "quality" / "audit.json"
    quality_ppl_pass = False
    quality_zero_shot_pass = False
    quality_all_models_ppl_pass = False
    if quality_audit_path.exists():
        quality_audit = load(quality_audit_path)
        quality_ppl_pass = bool(
            quality_audit.get("ppl_pass", quality_audit.get("pass"))
        )
        quality_zero_shot_pass = bool(quality_audit.get("zero_shot_pass", False))
        quality_all_models_ppl_pass = bool(
            quality_audit.get("paper_all_models_ppl_pass", False)
        )
    cacti_path = arguments.results / "cacti_memory.json"
    cacti_ok = False
    cacti_document = None
    if cacti_path.exists():
        cacti_document = load(cacti_path)
        cacti_ok = cacti_document.get("evidence") == "measured_open_proxy"
    drampower_path = arguments.results / "drampower_lpddr5.json"
    drampower_ok = False
    if drampower_path.exists():
        drampower = load(drampower_path)
        drampower_ok = (
            drampower.get("evidence") == "measured_open_proxy"
            and float(drampower.get("mean_energy_pj_per_byte", 0)) > 0
        )
    chipyard_scala_path = arguments.results / "chipyard_scala_compile.json"
    chipyard_scala_ok = False
    if chipyard_scala_path.exists():
        chipyard_scala = load(chipyard_scala_path)
        chipyard_scala_ok = (
            chipyard_scala.get("evidence") == "measured_native_sbt_compile"
            and bool(chipyard_scala.get("pass"))
        )
    checks: List[Dict[str, Any]] = []
    if quality_audit_path.exists():
        checks.extend(
            [
                {
                    "name": "oasis.full_ppl.opt_mistral",
                    "pass": quality_ppl_pass,
                    "evidence": "measured_full_token_quality_audit",
                },
                {
                    "name": "oasis.full_zero_shot.mistral",
                    "pass": quality_zero_shot_pass,
                    "evidence": "measured_full_lm_eval_quality_audit",
                },
            ]
        )
    if ir_contract is not None:
        checks.append(
            {
                "name": "compiler.serialized_ir_cycle_contract",
                "pass": all(
                    item.get("compiler_simulator_contract") == "validated"
                    and len(item.get("ir_sha256", "")) == 64
                    and int(item.get("cycles", 0)) > 0
                    for item in ir_contract.get("programs", {}).values()
                )
                and len(ir_contract.get("programs", {})) == 3,
                "evidence": "measured_serialized_ir_round_trip",
            }
        )

    pipeline_target = oasis_target["pipeline_1x4096x4096_w4a4_1pct"]
    ledger = anchor["metadata"]["cycle_ledger"]
    mapping = {
        "activation_quantization": "activation_quantization_cycles",
        "concat": "concat_cycles",
        "index_distribution": "index_distribution_cycles",
        "reduction": "reduction_cycles",
        "outlier_detection": "outlier_detection_cycles",
        "error_compensation": "error_compensation_cycles",
        "merge": "merge_cycles",
        "total": "critical_total_cycles",
    }
    for measured_name, target_name in mapping.items():
        relative_check(
            checks,
            "oasis.pipeline.%s" % measured_name,
            float(ledger[measured_name]),
            float(pipeline_target[target_name]),
            evidence="measured_from_calibrated_cycle_model",
        )

    traffic_target = oasis_target["memory_traffic_percent_1x4096x4096_w4a4_1pct"]
    traffic = anchor["metadata"]["sram_traffic_percent"]
    for name in ("weight_index_buffer", "lut", "activation_index_buffer", "output_buffer"):
        absolute_check(
            checks,
            "oasis.sram_traffic.%s" % name,
            float(traffic[name]),
            float(traffic_target[name]),
            0.25,
            "percentage_point",
        )

    energy_target = oasis_target["energy_percent_1x4096x4096_w4a4_1pct"]
    stage_energy = anchor["metadata"]["stage_energy_percent"]
    for name in (
        "activation_quantization",
        "index_distribution",
        "error_compensation",
        "concat",
        "reduction",
        "outlier_detection",
        "merge",
    ):
        target = energy_target[name]
        absolute_check(
            checks,
            "oasis.stage_energy.%s" % name,
            float(stage_energy[name]),
            float(target),
            0.25,
            "percentage_point",
        )

    headlines = oasis_target["single_batch_decode_geomean"]
    for precision in ("a4", "a3"):
        relative_check(
            checks,
            "oasis.%s.speedup_vs_figlut" % precision,
            float(system[precision]["geomean_speedup"]),
            float(headlines["%s_speedup_vs_figlut" % precision]),
            evidence="predicted_vs_inferred_figlut",
        )
        relative_check(
            checks,
            "oasis.%s.energy_efficiency_vs_figlut" % precision,
            float(system[precision]["geomean_energy_efficiency_improvement"]),
            float(headlines["%s_energy_efficiency_vs_figlut" % precision]),
            evidence="predicted_vs_inferred_figlut",
        )
        if gpu_system is not None:
            for baseline, target_prefix in (
                ("a100_fp16", "a100"),
                ("quarot_w4a4", "quarot"),
            ):
                measured = gpu_system[precision]["geomeans"][baseline]
                relative_check(
                    checks,
                    "oasis.%s.speedup_vs_%s" % (precision, target_prefix),
                    float(measured["speedup"]),
                    float(headlines["%s_speedup_vs_%s" % (precision, target_prefix)]),
                    evidence="predicted_vs_calibrated_roofline_not_native",
                )
                relative_check(
                    checks,
                    "oasis.%s.energy_efficiency_vs_%s"
                    % (precision, target_prefix),
                    float(measured["energy_efficiency_improvement"]),
                    float(
                        headlines[
                            "%s_energy_efficiency_vs_%s"
                            % (precision, target_prefix)
                        ]
                    ),
                    evidence="predicted_vs_calibrated_roofline_not_native",
                )
        if lookahead is not None:
            relative_check(
                checks,
                "oasis.%s.lookahead_throughput_improvement" % precision,
                float(lookahead[precision]["throughput_improvement"]),
                float(
                    oasis_target["other_reported_headlines"][
                        "lookahead_%s_throughput_gain_at_1pct" % precision
                    ]
                ),
                evidence="predicted_mechanism_ablation",
            )

    if cacti_ok and cacti_document is not None:
        relative_check(
            checks,
            "oasis.sram_area_cacti32_proxy",
            float(cacti_document["oasis_total_sram_area_mm2"]),
            float(oasis_target["hardware"]["reported_sram_area_mm2"]),
            evidence="measured_cacti32_proxy_vs_reported28nm",
        )

    absolute_check(
        checks,
        "oasis.functional.lookahead_identity",
        float(functional["oasis"]["maximum_identity_error"]),
        0.0,
        1e-4,
        "absolute_fp32",
    )
    checks.append(
        {
            "name": "omni.quantization_compensation_non_regression",
            "pass": bool(summary["omni_compensation_improves_rmse"]),
            "evidence": "measured_functional",
        }
    )
    omni_reported = omni_target["reported"]
    relative_check(
        checks,
        "omni.energy_efficiency_range.minimum",
        float(summary["omni_energy_efficiency_min"]),
        float(omni_reported["energy_efficiency_improvement_vs_sota_lut_min"]),
        evidence="predicted_from_inferred_28nm_coefficients",
    )
    relative_check(
        checks,
        "omni.energy_efficiency_range.maximum",
        float(summary["omni_energy_efficiency_max"]),
        float(omni_reported["energy_efficiency_improvement_vs_sota_lut_max"]),
        evidence="predicted_from_inferred_28nm_coefficients",
    )

    coverage = {
        "oasis_functional_core": True,
        "compiler_simulator_ir_contract": ir_contract is not None,
        "oasis_cycle_anchor": True,
        "oasis_figlut_headline": True,
        "oasis_paper_all_models_full_ppl": quality_all_models_ppl_pass,
        "oasis_opt_mistral_full_ppl": quality_ppl_pass,
        "oasis_mistral_full_zero_shot": quality_zero_shot_pass,
        "oasis_a100_and_quarot_native_baselines": False,
        "oasis_a100_and_quarot_calibrated_rooflines": gpu_system is not None,
        "oasis_lookahead_ablation": lookahead is not None,
        "omni_functional_core": True,
        "omni_context_cycle_sweep": True,
        "omni_reported_1p25_to_1p91_energy_range": bool(
            summary["omni_headline_numeric_validation_available"]
        ),
        "open_rtl_synthesis_proxy": rtl_synthesis_ok,
        "open_cacti_sram_proxy": cacti_ok,
        "open_drampower_lpddr5_proxy": drampower_ok,
        "full_rtl_pnr_ppa": False,
        "chipyard_full_soc_boot": False,
        "chipyard_scala_overlay_compile": chipyard_scala_ok,
    }
    available_pass = all(check["pass"] for check in checks)
    complete = available_pass and all(coverage.values())
    document = {
        "available_checks_pass": available_pass,
        "coverage_complete": all(coverage.values()),
        "complete": complete,
        "checks": checks,
        "coverage": coverage,
        "policy": "pointwise relative error <= 10%; traffic plot uses 0.25 percentage-point tolerance",
    }
    output = arguments.output or arguments.results / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))
    if not available_pass or (arguments.strict and not complete):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

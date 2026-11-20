"""Sweep representative LLM GEMMs and persist PRISM DSE/cost-model results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from costmodel import GemmShape, PrecisionSpec
from dse import DseRequest, NoFeasibleDesignError, PrismDse, ResourceConstraints, SearchSpace
from generator import HardwareGenerator


@dataclass(frozen=True)
class OperatorSpec:
    model: str
    phase: str
    operator: str
    m: int
    n: int
    k: int


MODEL_DIMS = {
    "llama3_8b": (4096, 14336),
    "llama3_70b": (8192, 28672),
    "qwen2_5_7b": (3584, 18944),
    "mixtral_8x7b": (4096, 14336),
}


def mainstream_llm_gemms(models: Sequence[str], phases: Sequence[str], prefill_tokens: int) -> list[OperatorSpec]:
    unknown = set(models) - set(MODEL_DIMS)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    workloads = []
    for model in models:
        hidden, intermediate = MODEL_DIMS[model]
        for phase in phases:
            if phase not in {"decode", "prefill"}:
                raise ValueError("phases must be decode and/or prefill")
            m = 1 if phase == "decode" else prefill_tokens
            workloads.extend(
                [
                    OperatorSpec(model, phase, "qkv_projection", m, 3 * hidden, hidden),
                    OperatorSpec(model, phase, "attention_output", m, hidden, hidden),
                    OperatorSpec(model, phase, "ffn_gate_up", m, 2 * intermediate, hidden),
                    OperatorSpec(model, phase, "ffn_down", m, hidden, intermediate),
                ]
            )
    return workloads


def parse_precision(label: str) -> tuple[PrecisionSpec, PrecisionSpec, PrecisionSpec]:
    match = re.fullmatch(r"W(\d+)A(\d+)(?:O(\d+))?", label.upper())
    if match is None:
        raise ValueError("precision must use W<bits>A<bits> or W<bits>A<bits>O<bits>, e.g. W4A8O8")
    weight_bits, activation_bits, output_bits = (int(value) if value else 8 for value in match.groups())
    return PrecisionSpec(weight_bits), PrecisionSpec(activation_bits), PrecisionSpec(output_bits)


def experiment_search_space(quick: bool) -> SearchSpace:
    if quick:
        return SearchSpace(
            array_rows=(8, 16, 32), array_cols=(8, 16, 32), tile_ks=(16, 32),
            lut_group_sizes=(1,), lut_digit_bits=(4,), lut_sharing=("global",), hybrid_lut_fractions=(0.5,)
        )
    return SearchSpace(
        array_rows=(8, 16, 32, 64), array_cols=(8, 16, 32, 64), tile_ks=(16, 32, 64),
        lut_group_sizes=(1, 2), lut_digit_bits=(2, 4), lut_sharing=("global", "per_tile"),
        hybrid_lut_fractions=(0.25, 0.5, 0.75),
    )


def run_sweep(
    workloads: Iterable[OperatorSpec], precision_labels: Sequence[str], output_dir: str | Path,
    search_space: SearchSpace, constraints: ResourceConstraints | None = None, emit_rtl: bool = False,
) -> list[dict[str, object]]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    dse, generator = PrismDse(), HardwareGenerator()
    records: list[dict[str, object]] = []
    for workload in workloads:
        for label in precision_labels:
            weight, activation, output = parse_precision(label)
            request = DseRequest(GemmShape(workload.m, workload.n, workload.k), weight, activation, output, constraints or ResourceConstraints(), search_space=search_space)
            record: dict[str, object] = {"workload": asdict(workload), "precision": label}
            try:
                result = dse.search(request)
                record.update({"status": "ok", "result": result.as_dict()})
                if emit_rtl:
                    name = f"{workload.model}_{workload.phase}_{workload.operator}_{label}".lower()
                    record["generated"] = {key: str(value) if value else None for key, value in asdict(generator.generate(request, result, target / "rtl" / name, name)).items()}
            except NoFeasibleDesignError as error:
                record.update({"status": "infeasible", "error": str(error)})
            records.append(record)
    _write_results(target, records)
    return records


def _write_results(target: Path, records: Sequence[dict[str, object]]) -> None:
    (target / "dse_reports.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="ascii")
    fields = ["status", "model", "phase", "operator", "m", "n", "k", "precision", "primitive", "rows", "cols", "tile_k", "dataflow", "lut_group_size", "lut_digit_bits", "lut_sharing", "lut_fraction", "accumulator_bits", "area_mm2", "energy_pj", "latency_cycles", "lut_storage_bits", "utilization", "pareto_candidates", "error"]
    with (target / "dse_summary.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            workload = record["workload"]
            row = {"status": record["status"], "precision": record["precision"], **workload}
            if record["status"] == "ok":
                result = record["result"]
                config, report, summary = result["selected_accelerator"], result["cost_report"], result["search_summary"]
                lut = config["lut"] or {}
                row.update({"primitive": config["primitive"], "rows": config["array"]["rows"], "cols": config["array"]["cols"], "tile_k": config["array"]["tile_k"], "dataflow": config["array"]["dataflow"], "lut_group_size": lut.get("group_size"), "lut_digit_bits": lut.get("digit_bits"), "lut_sharing": lut.get("sharing"), "lut_fraction": config["lut_fraction"], "accumulator_bits": config["accumulator_precision"]["bits"], "area_mm2": report["area_mm2"], "energy_pj": report["energy_pj"], "latency_cycles": report["latency_cycles"], "lut_storage_bits": report["lut_storage_bits"], "utilization": report["utilization"], "pareto_candidates": summary["pareto_candidates"]})
            else:
                row["error"] = record["error"]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results", help="directory for CSV and JSONL results")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_DIMS), default=["llama3_8b", "qwen2_5_7b"])
    parser.add_argument("--phases", nargs="+", choices=("decode", "prefill"), default=["decode", "prefill"])
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--precisions", nargs="+", default=["W4A8O8", "W4A16O16", "W8A8O8", "W8A16O16"])
    parser.add_argument("--full", action="store_true", help="use the larger candidate space")
    parser.add_argument("--emit-rtl", action="store_true")
    parser.add_argument("--max-area-mm2", type=float)
    parser.add_argument("--max-energy-pj", type=float)
    parser.add_argument("--max-latency-cycles", type=int)
    args = parser.parse_args()
    workloads = mainstream_llm_gemms(args.models, args.phases, args.prefill_tokens)
    constraints = ResourceConstraints(args.max_area_mm2, args.max_energy_pj, args.max_latency_cycles)
    records = run_sweep(workloads, args.precisions, args.output, experiment_search_space(not args.full), constraints, args.emit_rtl)
    print(f"wrote {len(records)} workload/precision records to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

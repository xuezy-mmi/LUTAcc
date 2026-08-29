#!/usr/bin/env python3
"""Open Yosys generic-gate synthesis and cross-design 28-nm proxy scaling."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = Path(os.environ.get("LUTACC_RTL_TOOL_ROOT", "/data3/docker_cache/LUTAcc/tools"))
YOSYS_ROOT = TOOL_ROOT / "yosys-0.7"
YOSYS = YOSYS_ROOT / "usr" / "bin" / "yosys"


MODULES = {
    "oasis_concat": ["rtl/oasis/oasis_concat.sv"],
    "oasis_cluster": ["rtl/oasis/oasis_cluster.sv"],
    "orizuru_tree": ["rtl/oasis/orizuru_tree.sv"],
    "oasis_index_counter": ["rtl/oasis/oasis_index_counter.sv"],
    "oasis_mac_tree": ["rtl/oasis/oasis_mac_tree.sv"],
    "oasis_compensation_mac": ["rtl/oasis/oasis_compensation_mac.sv"],
    "oasis_controller": ["rtl/oasis/oasis_controller.sv"],
    "omni_bqu": ["rtl/omni/omni_bqu.sv"],
    "omni_lgu": ["rtl/omni/omni_lgu.sv"],
    "omni_lut_pe": ["rtl/omni/omni_lut_pe.sv"],
    "omni_controller": ["rtl/omni/omni_controller.sv"],
}


def parse_stats(log: str, top: str) -> Dict[str, Any]:
    marker = "=== %s ===" % top
    position = log.rfind(marker)
    if position < 0:
        raise RuntimeError("Yosys did not report statistics for %s" % top)
    section = log[position:]
    total_match = re.search(r"Number of cells:\s+(\d+)", section)
    if not total_match:
        raise RuntimeError("Yosys cell total missing for %s" % top)
    cells: Dict[str, int] = {}
    in_cells = False
    for line in section.splitlines():
        if "Number of cells:" in line:
            in_cells = True
            continue
        if in_cells:
            match = re.match(r"\s+(\S+)\s+(\d+)\s*$", line)
            if match:
                cells[match.group(1)] = int(match.group(2))
            elif line.strip() == "":
                break
    return {"generic_cells": int(total_match.group(1)), "cell_types": cells}


def synthesize(top: str, sources: List[str]) -> Dict[str, Any]:
    if not YOSYS.exists():
        raise RuntimeError("run scripts/bootstrap_rtl_tools.sh first")
    command = (
        "read_verilog -sv %s; hierarchy -check -top %s; "
        "proc; opt; memory; opt; techmap; opt; stat"
        % (" ".join(sources), top)
    )
    environment = dict(os.environ)
    library = str(YOSYS_ROOT / "usr" / "lib" / "x86_64-linux-gnu")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else ""
    )
    environment["YOSYS_DATDIR"] = str(YOSYS_ROOT / "usr" / "share" / "yosys")
    completed = subprocess.run(
        [str(YOSYS), "-Q", "-p", command],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    result = parse_stats(completed.stdout, top)
    result["sources"] = sources
    return result


def scaled_proxy(modules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cell = lambda name: modules[name]["generic_cells"]
    oasis_instances = {
        "oasis_concat": 4096 * 16,
        "oasis_index_counter": 32 * 16,
        "oasis_mac_tree": 16,
        "oasis_compensation_mac": 16,
        "orizuru_tree": 273,
        "oasis_cluster": 4,
    }
    oasis_cells = sum(cell(name) * count for name, count in oasis_instances.items())
    # Covered paper area excludes buffers, dequantizer, functional unit, and
    # memory controller because this RTL subset does not implement those macros.
    covered_reported_area_mm2 = (
        (0.0868 + 0.271 + 0.117 + 0.0226) * 16 + 0.739 + 0.00131
    )
    mm2_per_generic_gate = covered_reported_area_mm2 / oasis_cells
    omni_instances = {
        "omni_lut_pe": 32 * 4,
        "omni_bqu": 1,
        "omni_lgu": 1,
        "omni_controller": 1,
    }
    omni_cells = sum(cell(name) * count for name, count in omni_instances.items())
    return {
        "oasis": {
            "instances": oasis_instances,
            "scaled_generic_cells": oasis_cells,
            "covered_reported_area_mm2": covered_reported_area_mm2,
        },
        "calibration": {
            "mm2_per_generic_gate": mm2_per_generic_gate,
            "technology_nm": 28,
            "evidence": "calibrated",
            "warning": "Generic techmapped gates are not standard-cell P&R area.",
        },
        "omni_lut": {
            "instances": omni_instances,
            "scaled_generic_cells": omni_cells,
            "predicted_logic_area_mm2": omni_cells * mm2_per_generic_gate,
            "evidence": "predicted_from_oasis_generic_gate_calibration",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "results" / "rtl_synthesis.json")
    arguments = parser.parse_args()
    modules = {name: synthesize(name, sources) for name, sources in MODULES.items()}
    document = {
        "tool": subprocess.check_output(
            [str(YOSYS), "-V"],
            env={
                **os.environ,
                "LD_LIBRARY_PATH": str(YOSYS_ROOT / "usr" / "lib" / "x86_64-linux-gnu"),
            },
            text=True,
        ).strip(),
        "modules": modules,
        "scaled_proxy": scaled_proxy(modules),
        "ppa_claim": "open_generic_gate_proxy_only",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

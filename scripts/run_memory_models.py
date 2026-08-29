#!/usr/bin/env python3
"""Run CACTI 7 at its nearest supported node (32 nm) for all SRAM blocks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = Path(os.environ.get("LUTACC_MEMORY_TOOL_ROOT", "/data3/docker_cache/LUTAcc/tools"))
CACTI_ROOT = TOOL_ROOT / "cacti"
CACTI = CACTI_ROOT / "cacti"


MEMORIES = {
    # Effective ports follow the concurrent producers/consumers in the paper
    # datapath (double-buffered W/LUT, 4 cluster writers + 2 readers, and eight
    # output clients). CACTI accounts for the physical multi-port overhead.
    "oasis_weight_index_per_line": {
        "bytes": 2048, "banks": 1, "instances": 16, "line_bytes": 64,
        "rw_ports": 2
    },
    "oasis_output": {
        "bytes": 65536, "banks": 1, "instances": 1, "line_bytes": 64,
        "rw_ports": 8
    },
    "oasis_activation_index": {
        "bytes": 16384, "banks": 1, "instances": 1, "line_bytes": 64,
        "rw_ports": 6
    },
    "oasis_lut": {
        "bytes": 2048, "banks": 1, "instances": 1, "line_bytes": 64,
        "rw_ports": 2
    },
    "omni_kv_sram": {
        "bytes": 2 * 1024 * 1024, "banks": 32, "instances": 1, "line_bytes": 64,
        "rw_ports": 1
    },
    "omni_lut_sram": {
        "bytes": 256 * 1024, "banks": 4, "instances": 1, "line_bytes": 64,
        "rw_ports": 1
    },
}


def replace_line(text: str, prefix: str, replacement: str) -> str:
    pattern = re.compile(r"^" + re.escape(prefix) + r".*$", re.MULTILINE)
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("CACTI template key missing: %s" % prefix)
    return updated


def parse_number(output: str, label: str) -> float:
    match = re.search(re.escape(label) + r"\s*:\s*([-+0-9.eE]+)", output)
    if not match:
        raise RuntimeError("CACTI output field missing: %s" % label)
    return float(match.group(1))


def run_cacti(name: str, specification: Dict[str, int]) -> Dict[str, Any]:
    template = (CACTI_ROOT / "cache.cfg").read_text(encoding="utf-8")
    config = replace_line(template, "-size (bytes)", "-size (bytes) %d" % specification["bytes"])
    config = replace_line(
        config,
        "-block size (bytes)",
        "-block size (bytes) %d" % specification["line_bytes"],
    )
    config = replace_line(config, "-associativity", "-associativity 1")
    config = replace_line(
        config,
        "-read-write port",
        "-read-write port %d" % specification["rw_ports"],
    )
    config = replace_line(config, "-UCA bank count", "-UCA bank count %d" % specification["banks"])
    config = replace_line(config, "-technology (u)", "-technology (u) 0.032")
    config = replace_line(
        config,
        "-output/input bus width",
        "-output/input bus width %d" % (specification["line_bytes"] * 8),
    )
    config = replace_line(config, '-cache type "cache"', '-cache type "ram"')
    with tempfile.TemporaryDirectory(prefix="lutacc-cacti-") as directory:
        path = Path(directory) / (name + ".cfg")
        path.write_text(config, encoding="utf-8")
        completed = subprocess.run(
            [str(CACTI), "-infile", str(path)],
            cwd=CACTI_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
    output = completed.stdout
    dimensions = re.search(
        r"Cache height x width \(mm\):\s*([-+0-9.eE]+)\s*x\s*([-+0-9.eE]+)",
        output,
    )
    if not dimensions:
        raise RuntimeError("CACTI dimensions missing for %s" % name)
    height = float(dimensions.group(1))
    width = float(dimensions.group(2))
    instances = specification["instances"]
    return {
        **specification,
        "technology_nm": 32,
        "access_time_ns": parse_number(output, "Access time (ns)"),
        "cycle_time_ns": parse_number(output, "Cycle time (ns)"),
        "read_energy_nj": parse_number(
            output, "Total dynamic read energy per access (nJ)"
        ),
        "write_energy_nj": parse_number(
            output, "Total dynamic write energy per access (nJ)"
        ),
        "leakage_power_mw": parse_number(
            output, "Total leakage power of a bank (mW)"
        ),
        "height_mm": height,
        "width_mm": width,
        "area_mm2_per_instance": height * width,
        "area_mm2_all_instances": height * width * instances,
        "evidence": "measured_cacti32_proxy",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "results" / "cacti_memory.json",
    )
    arguments = parser.parse_args()
    if not CACTI.exists():
        raise SystemExit("CACTI executable missing; run the memory-tool bootstrap")
    results = {name: run_cacti(name, spec) for name, spec in MEMORIES.items()}
    oasis_names = [name for name in results if name.startswith("oasis_")]
    omni_names = [name for name in results if name.startswith("omni_")]
    document = {
        "tool": "HewlettPackard/cacti",
        "tool_commit": subprocess.check_output(
            ["git", "-C", str(CACTI_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "technology_nm": 32,
        "technology_note": "CACTI 7 supports 32 nm as the nearest public proxy to paper 28 nm.",
        "memories": results,
        "oasis_total_sram_area_mm2": sum(
            results[name]["area_mm2_all_instances"] for name in oasis_names
        ),
        "omni_total_sram_area_mm2": sum(
            results[name]["area_mm2_all_instances"] for name in omni_names
        ),
        "evidence": "measured_open_proxy",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

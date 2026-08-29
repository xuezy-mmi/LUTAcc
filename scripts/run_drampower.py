#!/usr/bin/env python3
"""Derive LPDDR5 streaming read/write energy with DRAMPower 6.0.0."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = Path(os.environ.get("LUTACC_MEMORY_TOOL_ROOT", "/data3/docker_cache/LUTAcc/tools"))
DRAMPOWER_ROOT = TOOL_ROOT / "DRAMPower"
CLI = DRAMPOWER_ROOT / "build" / "bin" / "cli"
CONFIG = DRAMPOWER_ROOT / "tests" / "tests_drampower" / "resources" / "cliconfig.json"
MEMSPEC = DRAMPOWER_ROOT / "tests" / "tests_drampower" / "resources" / "lpddr5.json"


def command_trace(command: str, commands: int) -> str:
    lines = ["0,ACT,0,0,0,0,0"]
    for index in range(commands):
        lines.append(
            "%d,%s,0,0,0,%d,0,0x0102030405060708"
            % (15 + index * 5, command, index % 1024)
        )
    finish = 15 + commands * 5 + 20
    lines.append("%d,PRE,0,0,0,0,0" % finish)
    lines.append("%d,END,0,0,0,0,0" % (finish + 5))
    return "\n".join(lines) + "\n"


def run(command: str, commands: int) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
        handle.write(command_trace(command, commands))
        trace = Path(handle.name)
    try:
        output = subprocess.check_output(
            [str(CLI), "-c", str(CONFIG), "-m", str(MEMSPEC), "-t", str(trace)],
            cwd=DRAMPOWER_ROOT,
            text=True,
        )
    finally:
        trace.unlink(missing_ok=True)
    match = re.search(r"Total Energy ->\s*([-+0-9.eE]+)", output)
    if not match:
        raise RuntimeError("DRAMPower total energy missing")
    energy_j = float(match.group(1))
    # The bundled LPDDR5 x8 / burstLength=8 memspec transfers 64 bits/command.
    transferred_bytes = commands * 8
    return {
        "command": command,
        "commands": commands,
        "bytes_per_command": 8,
        "transferred_bytes": transferred_bytes,
        "total_energy_j": energy_j,
        "energy_pj_per_byte": energy_j * 1e12 / transferred_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "results" / "drampower_lpddr5.json",
    )
    arguments = parser.parse_args()
    if arguments.commands <= 0:
        parser.error("commands must be positive")
    if not CLI.exists():
        raise SystemExit("DRAMPower CLI missing; build v6.0.0 first")
    read = run("RD", arguments.commands)
    write = run("WR", arguments.commands)
    document = {
        "tool": "tukl-msd/DRAMPower",
        "version": "v6.0.0",
        "commit": subprocess.check_output(
            ["git", "-C", str(DRAMPOWER_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "memspec": "bundled LPDDR5 x8 burstLength=8",
        "read": read,
        "write": write,
        "mean_energy_pj_per_byte": (
            read["energy_pj_per_byte"] + write["energy_pj_per_byte"]
        )
        / 2,
        "timing_note": "DRAMPower computes energy and does not validate trace timing.",
        "evidence": "measured_open_proxy",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compile the LUTAcc LazyRoCC overlay against the local Chipyard tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chipyard", type=Path, default=Path("/root/chipyard"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "results" / "chipyard_scala_compile.json",
    )
    arguments = parser.parse_args()
    repository_file = ROOT / "integration" / "chipyard" / "sbt-repositories"
    source_directory = ROOT / "integration" / "chipyard" / "src" / "main" / "scala"
    command = [
        "sbt",
        "-Dsbt.override.build.repos=true",
        "-Dsbt.repository.config=%s" % repository_file,
        'set ProjectRef(uri("file:/root/chipyard/tools/chisel3/"), "chisel") / libraryDependencies ~= (_.filterNot(_.name == "treadle"))',
        "project rocketchip",
        'set Compile / unmanagedSourceDirectories += file("%s")' % source_directory,
        "compile",
    ]
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=arguments.chipyard,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - start
    output_tail = "\n".join(completed.stdout.splitlines()[-80:])
    commit = subprocess.check_output(
        ["git", "-C", str(arguments.chipyard), "rev-parse", "HEAD"], text=True
    ).strip()
    document = {
        "pass": completed.returncode == 0,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "chipyard_commit": commit,
        "project": "rocketchip",
        "source_directory": str(source_directory),
        "test_dependency_override": "removed Chisel test-only treadle SNAPSHOT",
        "output_tail": output_tail,
        "evidence": "measured_native_sbt_compile",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(output_tail)
    print(json.dumps({key: value for key, value in document.items() if key != "output_tail"}, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

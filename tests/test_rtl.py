import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="Verilator unavailable")
def test_rtl_lint_and_transaction_smoke():
    environment = dict(os.environ)
    environment["RTL_BUILD_DIR"] = str(ROOT / "build" / "rtl-pytest")
    subprocess.run([str(ROOT / "scripts" / "lint_rtl.sh")], cwd=ROOT, env=environment, check=True)
    subprocess.run(
        [str(ROOT / "scripts" / "run_rtl_smoke.sh")],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    environment["CHIPYARD_OVERLAY_BUILD_DIR"] = str(
        ROOT / "build" / "chipyard-overlay-pytest"
    )
    subprocess.run(
        [str(ROOT / "scripts" / "check_chipyard_overlay.sh")],
        cwd=ROOT,
        env=environment,
        check=True,
    )

"""Regenerate Experiment 1 plots from ``accuracy_samples.csv``."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from .accuracy import _plot_results
except ImportError:  # supports `python experiments/exp1/plot.py` from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from accuracy import _plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="experiments/exp1/results", help="directory containing accuracy_samples.csv and accuracy_summary.csv")
    args = parser.parse_args()
    directory = Path(args.input)
    with (directory / "accuracy_samples.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    with (directory / "accuracy_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    _plot_results(samples, summary, directory)
    print(f"updated {directory / 'error_boxplots.png'} and {directory / 'error_min_mean_max.png'}")


if __name__ == "__main__":
    main()

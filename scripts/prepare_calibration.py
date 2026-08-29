#!/usr/bin/env python3
"""Materialize the small C4 calibration subset with retries and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data3/docker_cache/LUTAcc/calibration/c4_validation_16.json"),
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.output.exists():
        payload = arguments.output.read_bytes()
        print(json.dumps({"path": str(arguments.output), "sha256": hashlib.sha256(payload).hexdigest(), "cached": True}))
        return 0
    last_error = None
    texts = []
    for attempt in range(arguments.retries):
        try:
            dataset = load_dataset(
                "allenai/c4", "en", split="validation", streaming=True
            )
            texts = []
            for item in dataset:
                text = item.get("text", "").strip()
                if text:
                    texts.append(text)
                if len(texts) >= arguments.samples:
                    break
            if len(texts) == arguments.samples:
                break
        except Exception as error:  # network transports vary across datasets/hub
            last_error = error
            time.sleep(min(30, 2 ** attempt))
    if len(texts) != arguments.samples:
        raise RuntimeError("could not materialize C4 calibration") from last_error
    document = {
        "dataset": "allenai/c4",
        "config": "en",
        "split": "validation",
        "samples": texts,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    arguments.output.write_bytes(payload)
    print(json.dumps({"path": str(arguments.output), "sha256": hashlib.sha256(payload).hexdigest(), "cached": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

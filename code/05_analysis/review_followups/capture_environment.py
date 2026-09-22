#!/usr/bin/env python3
"""Capture the execution environment before E4/E5 inference."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/fse2027_review_defense_e1_e8_20260730"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    battery = ROOT / "results/rq1_model_battery_3lang_20260723"
    grid = ROOT / "results/grounded_protocol_3lang_20260721"
    runners = {
        "battery": battery / "code/run_rq1_vlm.py",
        "grid": grid / "code/run_grounded_vlm.py",
        "rq3": grid / "code/run_rq3_vlm.py",
    }
    snapshot = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "transformers", "accelerate", "pandas")
        },
        "pillow": version("PIL"),
        "gpu_csv": command(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ).splitlines(),
        "cuda_runtime": __import__("torch").version.cuda,
        "cudnn_version": __import__("torch").backends.cudnn.version(),
        "runners": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in runners.items()
        },
        "prompt": {
            "path": str(grid / "config/prompt_B_template.txt"),
            "sha256": sha256(grid / "config/prompt_B_template.txt"),
        },
        "source_environment": {
            "path": str(battery / "audit/EXECUTION_ENVIRONMENT.json"),
            "sha256": sha256(battery / "audit/EXECUTION_ENVIRONMENT.json"),
        },
        "frozen_policy": {
            "seed": 42,
            "dtype": "bfloat16",
            "do_sample": False,
            "temperature": 0.0,
            "top_p_inactive": 1.0,
            "max_new_tokens": 24,
            "packaging": "two_separately_labeled_images",
        },
    }
    path = OUT / "env/PRE_EXECUTION_ENVIRONMENT.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()

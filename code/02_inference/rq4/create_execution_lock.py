#!/usr/bin/env python3
"""Create the immutable F5 execution-environment lock before inference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/deploy_v2_logit_20260731"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_info(name: str) -> dict[str, str]:
    module = __import__(name)
    return {
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(Path(module.__file__).resolve()),
    }


def main() -> None:
    import torch

    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if selected != "0":
        raise RuntimeError(f"F5 preregistered GPU is 0, got {selected!r}")
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpu_rows = {}
    for line in query.strip().splitlines():
        index, uuid, name, driver, memory = line.split(", ", 4)
        gpu_rows[index] = {
            "uuid": uuid,
            "name": name,
            "driver": driver,
            "memory_mib": memory,
        }
    packages = {
        name: module_info(name)
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "PIL",
            "pandas",
            "numpy",
            "scipy",
            "sklearn",
            "tokenizers",
            "safetensors",
        )
    }
    lock = {
        "run_id": "deploy_v2_logit_20260731",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_inference",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_visible_devices": selected,
        "gpu_uuid": gpu_rows[selected]["uuid"],
        "gpu": gpu_rows[selected],
        "batching": "batch_size_1_direct_generate_no_continuous_batching",
        "parallelism": {"tensor_parallel": 1, "pipeline_parallel": 1, "model_processes": 1},
        "precision": "bfloat16",
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p_inactive": 1.0,
            "max_new_tokens": 24,
            "seed": 42,
        },
        "environment": {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
        },
        "model_revisions": {
            "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        },
        "preregistration_sha256": sha256(OUT / "config/F5_PREREGISTRATION.md"),
        "frozen_inventory_sha256": sha256(OUT / "data/F5_FROZEN_INPUT_INVENTORY.csv"),
        "frozen_pairs_sha256": sha256(OUT / "data/F5_FROZEN_PAIRS.csv"),
        "frozen_items_sha256": sha256(OUT / "data/F5_FROZEN_ITEMS.csv"),
        "ocrml_gate_sha256": sha256(OUT / "data/F5_OCRML_GATE.json"),
        "runner_sha256": sha256(OUT / "code/run_f5_inference.py"),
        "logit_capture_source_sha256": sha256(
            ROOT / "results/protocol_unified_3lang_20260720/code/run_logit_pilot.py"
        ),
        "canonical_judge_source_sha256": sha256(
            ROOT / "experiments/rq0_viability/scripts/run_judge_pairs.py"
        ),
        "new_inference_budget": 54_000,
    }
    path = OUT / "config/F5_EXECUTION_LOCK.json"
    if path.exists():
        raise FileExistsError("execution lock already exists")
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

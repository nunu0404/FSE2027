#!/usr/bin/env python3
"""Validate embedding extraction shape, coverage, finiteness, and non-collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    stem = manifest_path.name.removesuffix(".manifest.json")
    metadata = pd.read_csv(manifest_path.with_name(stem + ".csv"))
    arrays = np.load(manifest_path.with_name(stem + ".npz"))
    expected_shapes = (
        {"native_patch_mean": (args.expected, 1280), "projected_visual_mean": (args.expected, 3584)}
        if manifest["model"].startswith("Qwen")
        else {"native_patch_mean": (args.expected, 1024), "projected_visual_mean": (args.expected, 3584),
              "native_cls": (args.expected, 1024)}
    )
    errors = []
    if manifest.get("status") != "complete" or len(metadata) != args.expected:
        errors.append("manifest or metadata row count incomplete")
    if set(arrays.files) != set(expected_shapes):
        errors.append(f"array key drift: {arrays.files}")
    diagnostics = {}
    for key, shape in expected_shapes.items():
        if key not in arrays:
            continue
        value = arrays[key].astype(np.float32)
        norms = np.linalg.norm(value, axis=1)
        diagnostics[key] = {
            "shape": list(value.shape), "finite": bool(np.isfinite(value).all()),
            "min_norm": float(norms.min()), "max_norm": float(norms.max()),
            "mean_feature_std": float(value.std(axis=0).mean()),
        }
        if value.shape != shape or not np.isfinite(value).all() or float(norms.min()) <= 0:
            errors.append(f"invalid array: {key}")
        if args.expected > 1 and float(value.std(axis=0).mean()) <= 1e-6:
            errors.append(f"collapsed embeddings: {key}")
    result = {
        "gate": "GD0" if args.expected < 9384 else "GD-full",
        "model": manifest.get("model"), "rows": len(metadata),
        "representation_conditions": int(metadata.representation_condition.nunique()),
        "diagnostics": diagnostics, "errors": errors, "gate_pass": not errors,
    }
    output = manifest_path.with_name(stem + ".validation.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

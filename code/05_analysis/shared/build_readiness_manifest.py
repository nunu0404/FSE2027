#!/usr/bin/env python3
"""Build a hash inventory and machine-readable readiness summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    core = [
        OUT / "config/protocol_grounded.json",
        OUT / "config/prompt_B_template.txt",
        OUT / "data/pairs_seed42_grounded.csv",
        OUT / "data/render_snippets.csv",
        OUT / "data/render_conditions.csv",
        OUT / "rendered/RENDER_MANIFEST.json",
        OUT / "audit/GROUNDED_RENDER_AUDIT.json",
        OUT / "audit/BLUR_OCR_MANIFEST.json",
        OUT / "audit/blur_ocr_summary.csv",
        OUT / "code/run_grounded_vlm.py",
        OUT / "EXPERIMENT_CONDITION_REGISTER.md",
        OUT / "GROUNDED_PROTOCOL_AMENDMENT.md",
        OUT / "RQ3_VARIANT_RULES_AND_GATE.md",
    ]
    missing = [str(path) for path in core if not path.is_file()]
    audit = json.loads((OUT / "audit/GROUNDED_RENDER_AUDIT.json").read_text())
    ocr = json.loads((OUT / "audit/BLUR_OCR_MANIFEST.json").read_text())
    pairs = pd.read_csv(OUT / "data/pairs_seed42_grounded.csv")
    dry_manifests = sorted((OUT / "inference").glob("*/raw/*dryrun*.manifest.json"))
    dry_pass = sum(json.loads(path.read_text()).get("status") == "dry_run_passed" for path in dry_manifests)
    result = {
        "protocol": "grounded-3lang-20260721-v2",
        "a_b_preparation_ready": not missing and audit.get("gate_pass") is True and ocr.get("status") == "complete" and dry_pass == 4,
        "rq3_inference_ready": False,
        "rq3_blocker": "variants must be generated, automatically audited, and pass author gate GC1",
        "missing_core_files": missing,
        "pairs": int(len(pairs)),
        "pairs_by_language": pairs.groupby("language").size().to_dict(),
        "render_gate": audit.get("gate_pass"),
        "ocr_gate": ocr.get("status"),
        "dry_run_manifests": len(dry_manifests),
        "dry_run_passed": dry_pass,
        "core_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in core if path.is_file()},
    }
    target = OUT / "READINESS_MANIFEST.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["a_b_preparation_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze source assets, code lineage, and deployment call count."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
BATTERY = ROOT / "results/rq1_model_battery_3lang_20260723"
PAIR_PATH = BATTERY / "data/rq1_pairs_9000.csv"
PROMPT_PATH = OUT / "config/FROZEN_PROMPT_B.txt"
DEPLOY_PAIRS = ROOT / "fse2027/external_runs/deploy_v2_logit_20260731/data/F5_FROZEN_PAIRS.csv"
SOURCE_FILES = {
    "predecessor_inference_runner": BATTERY / "code/run_rq1_vlm.py",
    "predecessor_analysis": BATTERY / "code/analyze_full_results.py",
    "legacy_verdict_logit_parser": ROOT / "results/protocol_unified_3lang_20260720/code/run_logit_pilot.py",
    "legacy_prompt_packaging": ROOT / "experiments/rq0_viability/scripts/run_judge_pairs.py",
    "new_inference_runner": OUT / "code/run_primary.py",
    "new_run_validator": OUT / "code/validate_primary_run.py",
    "new_primary_analysis": OUT / "code/analyze_primary.py",
    "stored_output_sanity": OUT / "code/sanity_check_existing.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    deploy = pd.read_csv(DEPLOY_PAIRS)
    if len(pairs) != 9000 or pairs.protocol_pair_id.nunique() != 9000:
        raise RuntimeError("primary manifest count drift")
    if len(deploy) != deploy.pair_id.nunique():
        raise RuntimeError("deployment manifest pair IDs are not unique")
    if deploy.groupby("language").size().to_dict() != {
        "cuda": 3000,
        "java": 3000,
        "python": 3000,
    }:
        raise RuntimeError("deployment language allocation drift")

    image_rows = pd.concat(
        [
            pairs[["image_i_path", "image_i_sha256"]].rename(
                columns={"image_i_path": "path", "image_i_sha256": "expected_sha256"}
            ),
            pairs[["image_j_path", "image_j_sha256"]].rename(
                columns={"image_j_path": "path", "image_j_sha256": "expected_sha256"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("path")
    image_rows["actual_sha256"] = image_rows.path.map(lambda path: sha256(ROOT / path))
    image_rows["match"] = image_rows.expected_sha256.eq(image_rows.actual_sha256)
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    image_rows.to_csv(audit_dir / "IMAGE_INTEGRITY.csv", index=False)

    source_inventory = []
    for role, path in SOURCE_FILES.items():
        source_inventory.append(
            {
                "role": role,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(source_inventory).to_csv(
        audit_dir / "CODE_LINEAGE.csv", index=False
    )
    battery_status = git(BATTERY, "status", "--porcelain")
    package_status = git(OUT, "status", "--porcelain")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "pre_inference_asset_and_lineage_audit",
        "historical_full_metric_reproduction": False,
        "primary": {
            "pair_manifest": str(PAIR_PATH.relative_to(ROOT)),
            "pair_manifest_sha256": sha256(PAIR_PATH),
            "pairs": len(pairs),
            "unique_pairs": pairs.protocol_pair_id.nunique(),
            "languages": pairs.groupby("language").size().to_dict(),
            "language_difficulty": {
                f"{language}|{difficulty}": int(count)
                for (language, difficulty), count in pairs.groupby(
                    ["language", "difficulty"]
                ).size().items()
            },
            "unique_pngs": len(image_rows),
            "image_hash_mismatches": int((~image_rows.match).sum()),
            "prompt": str(PROMPT_PATH.relative_to(ROOT)),
            "prompt_sha256": sha256(PROMPT_PATH),
        },
        "deployment": {
            "manifest": str(DEPLOY_PAIRS.relative_to(ROOT)),
            "manifest_sha256": sha256(DEPLOY_PAIRS),
            "pair_count": len(deploy),
            "expected_calls_formula": "2 * pair_count",
            "expected_calls": 2 * len(deploy),
        },
        "git": {
            "source_battery_commit": git(BATTERY, "rev-parse", "HEAD"),
            "source_battery_dirty": bool(battery_status),
            "source_battery_dirty_paths": battery_status.splitlines(),
            "new_package_commit": git(OUT, "rev-parse", "HEAD"),
            "new_package_dirty": bool(package_status),
            "new_package_dirty_paths": package_status.splitlines(),
        },
        "models": json.loads((OUT / "config/protocol.json").read_text())["models"],
        "generation": json.loads((OUT / "config/protocol.json").read_text())[
            "generation"
        ],
        "code_inventory": source_inventory,
    }
    report["gate_pass"] = bool(
        report["primary"]["pairs"] == 9000
        and report["primary"]["image_hash_mismatches"] == 0
        and report["deployment"]["expected_calls"] == 2 * report["deployment"]["pair_count"]
    )
    (audit_dir / "PRE_INFERENCE_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

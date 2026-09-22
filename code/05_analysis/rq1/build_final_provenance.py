#!/usr/bin/env python3
"""Build the final code/config/manifest provenance inventory."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    package_commit = subprocess.check_output(
        ["git", "-C", str(OUT), "rev-parse", "HEAD"], text=True
    ).strip()
    patterns = (
        "code/*",
        "config/*",
        "environments/*.json",
        "inference/*/*/manifest.json",
        "inference/*/*/validation.json",
        "analysis/*manifest.json",
        "analysis/*/*manifest.json",
    )
    paths = set()
    for pattern in patterns:
        paths.update(path for path in OUT.glob(pattern) if path.is_file())
    rows = [
        {
            "relative_path": str(path.relative_to(OUT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths)
    ]
    audit = OUT / "audit"
    audit.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(audit / "FINAL_PROVENANCE_INVENTORY.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_git_commit": package_commit,
        "package_git_status": subprocess.check_output(
            ["git", "-C", str(OUT), "status", "--porcelain"], text=True
        ).splitlines(),
        "inventory_entries": len(rows),
        "inference_code_versioning": "Each inference manifest records package_git_commit and runner_sha256.",
        "parser_versioning": "The verdict parser and logit extractor are embedded in the hashed inference runner.",
        "analysis_code_versioning": "Each analysis manifest records package_git_commit and analysis_script_sha256.",
    }
    (audit / "FINAL_PROVENANCE.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

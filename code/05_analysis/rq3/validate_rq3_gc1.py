#!/usr/bin/env python3
"""Validate the author-completed RQ3 GC1 form and authorize inference."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RQ3 = ROOT / "results/grounded_protocol_3lang_20260721/rq3"
FORM = RQ3 / "review/GC1_REVIEW_FORM.csv"
APPROVAL = RQ3 / "review/GC1_APPROVAL.json"
MANIFEST = RQ3 / "RQ3_PREPARATION_MANIFEST.json"
CHECKS = [
    "gold_coherent",
    "trash_incoherent",
    "clean_legible",
    "ugly_degraded_no_corruption",
    "same_tokens_clean_ugly",
    "approve",
]
AFFIRMATIVE = {"1", "approved", "pass", "true", "y", "yes"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(value: object) -> bool:
    return str(value).strip().lower() in AFFIRMATIVE


def main() -> int:
    frame = pd.read_csv(FORM, dtype=str, keep_default_na=False)
    errors: list[str] = []
    required = {"review_id", "base_id", "language", *CHECKS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    if len(frame) != 60:
        errors.append(f"expected 60 rows, found {len(frame)}")
    if "review_id" in frame and frame.review_id.duplicated().any():
        errors.append("duplicate review_id")
    if "base_id" in frame and frame.base_id.duplicated().any():
        errors.append("duplicate base_id")
    if "language" in frame:
        counts = frame.groupby("language").size().to_dict()
        if counts != {"cuda": 20, "java": 20, "python": 20}:
            errors.append(f"language allocation drift: {counts}")
    sample_path = RQ3 / "review/GC1_SAMPLE.csv"
    sample_columns = ["review_id", "base_id", "rq0_id", "language", "score_stratum"]
    if not sample_path.is_file():
        errors.append("frozen GC1_SAMPLE.csv is missing")
    elif not set(sample_columns).issubset(frame.columns):
        errors.append("review form lacks frozen-sample identity columns")
    else:
        sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        actual = frame[sample_columns].reset_index(drop=True)
        if not actual.equals(sample[sample_columns].reset_index(drop=True)):
            errors.append("review form identities/order do not match frozen GC1_SAMPLE.csv")
    failed_rows: list[dict[str, object]] = []
    if not missing_columns:
        for row in frame.itertuples(index=False):
            failed = [column for column in CHECKS if not accepted(getattr(row, column))]
            if failed:
                failed_rows.append({"review_id": row.review_id, "base_id": row.base_id, "failed": failed})
    if failed_rows:
        errors.append(f"{len(failed_rows)} rows are incomplete or not approved")

    automatic = json.loads((RQ3 / "audit/RQ3_AUTOMATIC_AUDIT.json").read_text(encoding="utf-8"))
    if not automatic.get("automatic_gate_pass"):
        errors.append("automatic RQ3 preparation gate is not passing")
    result = {
        "status": "approved" if not errors else "not_approved",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_form": str(FORM.relative_to(ROOT)),
        "review_form_sha256": sha256(FORM),
        "sample_sha256": sha256(sample_path) if sample_path.is_file() else None,
        "review_rows": len(frame),
        "checks_per_row": CHECKS,
        "failed_rows": failed_rows,
        "errors": errors,
        "gate_pass": not errors,
    }
    APPROVAL.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not errors:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest.update({
            "status": "ready_for_inference",
            "inference_authorized": True,
            "inference_blocker": None,
            "gc1_approval_sha256": sha256(APPROVAL),
            "gc1_approved_at_utc": result["validated_at_utc"],
        })
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

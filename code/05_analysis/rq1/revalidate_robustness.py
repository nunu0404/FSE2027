#!/usr/bin/env python3
"""Validate complete robustness runs while retaining model parse failures."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUT = Path("/ANON/experiment_root/results/latest_vlm_extension_20260830")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "internvl3_5"])
    args = parser.parse_args()
    run_dir = OUT / f"inference/robustness/{args.model}"
    raw = pd.read_json(run_dir / "raw.jsonl", lines=True)
    validation_path = run_dir / "validation.json"
    initial_path = run_dir / "validation_initial.json"
    if not initial_path.exists():
        shutil.copy2(validation_path, initial_path)
    initial = json.loads(initial_path.read_text())
    parsed = raw.parsed_choice.isin(["A", "B"])
    missing = raw.logit_A.isna() | raw.logit_B.isna() | raw.margin.isna()
    parsed_mismatch = int(
        (parsed & ~raw.argmax_matches_parsed.fillna(False).astype(bool)).sum()
    )
    structural_pass = bool(
        len(raw) == 7200
        and raw.pair_id.nunique() == 900
        and raw[["condition", "pair_id", "order"]].duplicated().sum() == 0
        and initial.get("missing_keys") == 0
        and initial.get("unexpected_keys") == 0
        and initial.get("mapping_errors") == 0
        and initial.get("condition_language_order_complete") is True
    )
    failures = raw.loc[~parsed].copy()
    failure_distribution = {
        "condition": failures.condition.value_counts().sort_index().to_dict(),
        "language": failures.language.value_counts().sort_index().to_dict(),
        "difficulty": failures.difficulty.value_counts().sort_index().to_dict(),
        "order": failures.order.value_counts().sort_index().to_dict(),
    }
    audit = {
        **initial,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_contract": "structural completeness plus parsed-call logit orientation; parse failures retained as strict-invalid model outcomes",
        "structural_gate_pass": structural_pass,
        "parse_failures": int((~parsed).sum()),
        "missing_logits": int(missing.sum()),
        "missing_logits_equal_parse_failures": int(missing.sum()) == int((~parsed).sum()),
        "argmax_mismatches_on_parsed_calls": parsed_mismatch,
        "parse_failures_reached_max_tokens": int(
            failures.reached_max_new_tokens.fillna(False).astype(bool).sum()
        ),
        "failure_distribution": failure_distribution,
        "raw_inference_modified": False,
    }
    audit["gate_pass"] = bool(
        structural_pass
        and audit["missing_logits_equal_parse_failures"]
        and parsed_mismatch == 0
    )
    validation_path.write_text(json.dumps(audit, indent=2) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = (
        "validated_complete"
        if audit["parse_failures"] == 0
        else "validated_complete_with_reported_parse_failures"
    )
    manifest["validation_path"] = str(validation_path.relative_to(OUT))
    manifest["raw_inference_modified"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["gate_pass"] else 1)


if __name__ == "__main__":
    main()

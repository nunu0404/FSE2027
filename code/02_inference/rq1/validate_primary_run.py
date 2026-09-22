#!/usr/bin/env python3
"""Validate a completed primary run before downstream stages are allowed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
PAIR_PATH = ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "internvl3_5", "gemma4"])
    args = parser.parse_args()
    run_dir = OUT / "inference/full" / args.model
    raw_path = run_dir / "raw.jsonl"
    manifest_path = run_dir / "manifest.json"
    raw = pd.read_json(raw_path, lines=True)
    pairs = pd.read_csv(PAIR_PATH)
    expected = pd.MultiIndex.from_product(
        [pairs.protocol_pair_id, ["AB", "BA"]], names=["pair_id", "order"]
    )
    actual = pd.MultiIndex.from_frame(raw[["pair_id", "order"]])
    duplicate_calls = int(raw[["pair_id", "order"]].duplicated().sum())
    missing_keys = len(expected.difference(actual))
    unexpected_keys = len(actual.difference(expected))
    parse_failures = int(raw.parsed_choice.isna().sum())
    missing_logits = int((raw.logit_A.isna() | raw.logit_B.isna() | raw.margin.isna()).sum())
    argmax_mismatches = int((~raw.argmax_matches_parsed.fillna(False).astype(bool)).sum())
    mapping_errors = int((~raw.image_mapping_verified.astype(bool)).sum())
    truncations = int(raw.reached_max_new_tokens.fillna(False).astype(bool).sum())
    exact = raw.raw_output.astype(str).str.fullmatch(r"FINAL_VERDICT: [AB]")
    non_exact_outputs = int((~exact).sum())
    language_order = (
        raw.groupby(["language", "order"]).size().sort_index().astype(int).to_dict()
    )
    language_order_json = {f"{key[0]}|{key[1]}": value for key, value in language_order.items()}
    expected_language_order = {
        f"{language}|{order}": 3000
        for language in ("java", "python", "cuda")
        for order in ("AB", "BA")
    }
    audit = {
        "model_key": args.model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_calls": 18000,
        "actual_calls": len(raw),
        "unique_pairs": raw.pair_id.nunique(),
        "duplicate_calls": duplicate_calls,
        "missing_pair_order_keys": missing_keys,
        "unexpected_pair_order_keys": unexpected_keys,
        "parse_failures": parse_failures,
        "missing_logits": missing_logits,
        "argmax_mismatches": argmax_mismatches,
        "mapping_errors": mapping_errors,
        "truncations": truncations,
        "non_exact_outputs": non_exact_outputs,
        "calls_by_language_order": language_order_json,
    }
    audit["gate_pass"] = bool(
        audit["actual_calls"] == 18000
        and audit["unique_pairs"] == 9000
        and language_order_json == expected_language_order
        and all(
            audit[key] == 0
            for key in (
                "duplicate_calls",
                "missing_pair_order_keys",
                "unexpected_pair_order_keys",
                "parse_failures",
                "missing_logits",
                "argmax_mismatches",
                "mapping_errors",
            )
        )
    )
    audit_path = run_dir / "validation.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "validated_complete" if audit["gate_pass"] else "validation_failed"
    manifest["validation_path"] = str(audit_path.relative_to(OUT))
    manifest["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

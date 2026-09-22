#!/usr/bin/env python3
"""Materialize model-effective verdict logits without altering raw inference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


OUT = Path("/ANON/experiment_root/results/latest_vlm_extension_20260830")
MODELS = ("qwen3", "internvl3_5", "gemma4")


def gemma_softcap(a: float, b: float) -> tuple[float, float]:
    values = torch.tensor([a, b], dtype=torch.bfloat16)
    values = torch.tanh(values / 30.0) * 30.0
    return float(values[0]), float(values[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    run_dir = OUT / f"inference/full/{args.model}"
    raw = pd.read_json(run_dir / "raw.jsonl", lines=True)
    rows = []
    for row in raw.itertuples(index=False):
        if pd.isna(row.logit_A) or pd.isna(row.logit_B):
            effective_a = effective_b = effective_margin = None
            effective_argmax = None
            matches = False
        else:
            if args.model == "gemma4":
                effective_a, effective_b = gemma_softcap(row.logit_A, row.logit_B)
                source = "lm_head_hook_plus_gemma4_bf16_30xtanh_logit_over_30"
            else:
                effective_a, effective_b = float(row.logit_A), float(row.logit_B)
                source = "lm_head_forward_hook"
            effective_margin = effective_a - effective_b
            effective_argmax = "A" if effective_a >= effective_b else "B"
            matches = effective_argmax == row.parsed_choice
        rows.append(
            {
                "pair_id": row.pair_id,
                "order": row.order,
                "parsed_choice": row.parsed_choice,
                "raw_logit_A": None if pd.isna(row.logit_A) else float(row.logit_A),
                "raw_logit_B": None if pd.isna(row.logit_B) else float(row.logit_B),
                "raw_margin": None if pd.isna(row.margin) else float(row.margin),
                "effective_logit_A": effective_a,
                "effective_logit_B": effective_b,
                "effective_margin": effective_margin,
                "effective_logit_tie": (
                    None if effective_margin is None else effective_margin == 0
                ),
                "effective_argmax_choice": effective_argmax,
                "effective_argmax_matches_parsed": matches,
                "effective_logit_source": (
                    "unavailable_parse_failure"
                    if effective_margin is None
                    else source
                ),
            }
        )
    effective = pd.DataFrame(rows)
    path = run_dir / "effective_logits.jsonl"
    effective.to_json(path, orient="records", lines=True)

    initial_path = run_dir / "validation_initial.json"
    validation_path = run_dir / "validation.json"
    if not initial_path.exists():
        shutil.copy2(validation_path, initial_path)
    initial = json.loads(initial_path.read_text())
    parsed = effective.parsed_choice.isin(["A", "B"])
    missing = effective.effective_margin.isna()
    effective_mismatch = int(
        (parsed & ~effective.effective_argmax_matches_parsed.astype(bool)).sum()
    )
    structural_keys = (
        "duplicate_calls",
        "missing_pair_order_keys",
        "unexpected_pair_order_keys",
        "mapping_errors",
    )
    structural_pass = bool(
        initial["actual_calls"] == initial["expected_calls"] == 18000
        and initial["unique_pairs"] == 9000
        and all(initial[key] == 0 for key in structural_keys)
    )
    parse_failures = int((~parsed).sum())
    missing_logits = int(missing.sum())
    audit = {
        **initial,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_contract": "structural completeness plus effective-logit orientation; parse failures are retained as model failures",
        "structural_gate_pass": structural_pass,
        "parse_failures": parse_failures,
        "missing_logits": missing_logits,
        "raw_argmax_mismatches": initial.get("argmax_mismatches"),
        "effective_argmax_mismatches_on_parsed_calls": effective_mismatch,
        "missing_logits_equal_parse_failures": missing_logits == parse_failures,
        "effective_logit_ties": int(
            effective.effective_logit_tie.fillna(False).astype(bool).sum()
        ),
        "effective_logit_path": str(path.relative_to(OUT)),
        "effective_logit_method": (
            "BF16 30*tanh(raw_lm_head_logit/30), matching Gemma4 model forward"
            if args.model == "gemma4"
            else "raw LM-head hook logits; no model-side final transform"
        ),
        "raw_inference_modified": False,
    }
    audit["gate_pass"] = bool(
        structural_pass
        and missing_logits == parse_failures
        and effective_mismatch == 0
    )
    validation_path.write_text(json.dumps(audit, indent=2) + "\n")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = (
        "validated_complete"
        if parse_failures == 0
        else "validated_complete_with_reported_parse_failures"
    )
    manifest["effective_logit_path"] = str(path.relative_to(OUT))
    manifest["effective_logit_validation"] = str(validation_path.relative_to(OUT))
    manifest["raw_inference_modified"] = False
    manifest["postprocess_package_git_commit"] = subprocess.check_output(
        ["git", "-C", str(OUT), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["gate_pass"] else 1)


if __name__ == "__main__":
    main()

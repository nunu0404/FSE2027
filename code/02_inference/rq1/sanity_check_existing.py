#!/usr/bin/env python3
"""Limited stored-output sanity check for the latest-VLM extension.

This intentionally checks a fixed 18-pair slice for two predecessor models. It
does not reproduce full historical metrics or run any model inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
BATTERY = ROOT / "results/rq1_model_battery_3lang_20260723"
PAIRS = BATTERY / "data/rq1_pairs_9000.csv"
RAW_FILES = {
    "qwen2_5": BATTERY / "inference/full/qwen/raw.jsonl",
    "internvl3": BATTERY / "inference/full/internvl/raw.jsonl",
}
VERDICT_RE = re.compile(
    r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B)\b", re.IGNORECASE
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_verdict(text: str) -> str | None:
    matches = list(VERDICT_RE.finditer(str(text or "")))
    return matches[-1].group(1).upper() if matches else None


def selected_snippet(row: pd.Series) -> str | None:
    if row.parsed_choice == "A":
        return row.snippet_first
    if row.parsed_choice == "B":
        return row.snippet_second
    return None


def fixed_sample(pairs: pd.DataFrame) -> pd.DataFrame:
    sample = (
        pairs.sort_values(["language", "difficulty", "protocol_pair_id"])
        .groupby(["language", "difficulty"], sort=True, group_keys=False)
        .head(2)
        .copy()
    )
    if len(sample) != 18 or sample.protocol_pair_id.nunique() != 18:
        raise RuntimeError("sanity sample must contain 18 unique pairs")
    return sample


def check_model(
    model_key: str, raw_path: Path, sample: pd.DataFrame
) -> tuple[dict, list[dict]]:
    raw = pd.read_json(raw_path, lines=True)
    raw = raw[raw.pair_id.isin(sample.protocol_pair_id)].copy()
    expected_keys = {
        (pair_id, order)
        for pair_id in sample.protocol_pair_id
        for order in ("AB", "BA")
    }
    actual_keys = set(zip(raw.pair_id, raw.order))
    duplicate_calls = int(raw[["pair_id", "order"]].duplicated().sum())

    raw["reparsed_choice"] = raw.raw_output.map(parse_verdict)
    parser_mismatches = int(
        (raw.reparsed_choice.fillna("NULL") != raw.parsed_choice.fillna("NULL")).sum()
    )
    raw["expected_first"] = raw.apply(
        lambda row: row.snippet_i if row.order == "AB" else row.snippet_j, axis=1
    )
    raw["expected_second"] = raw.apply(
        lambda row: row.snippet_j if row.order == "AB" else row.snippet_i, axis=1
    )
    mapping_errors = int(
        (
            raw.snippet_first.ne(raw.expected_first)
            | raw.snippet_second.ne(raw.expected_second)
            | ~raw.image_mapping_verified.astype(bool)
        ).sum()
    )
    missing_logits = int((raw.logit_A.isna() | raw.logit_B.isna()).sum())
    margin_errors = int(
        (~(raw.margin - (raw.logit_A - raw.logit_B)).abs().le(1e-9)).sum()
    )
    expected_argmax = raw.apply(
        lambda row: "A" if row.logit_A >= row.logit_B else "B", axis=1
    )
    argmax_errors = int(
        (
            raw.logit_argmax_choice.ne(expected_argmax)
            | raw.argmax_matches_parsed.ne(expected_argmax.eq(raw.parsed_choice))
        ).sum()
    )

    raw["selected_snippet_check"] = raw.apply(selected_snippet, axis=1)
    details = []
    bc_violations = 0
    boundary_pairs = 0
    content_ties = 0
    for pair_id, frame in raw.groupby("pair_id", sort=True):
        frame = frame.set_index("order")
        if set(frame.index) != {"AB", "BA"}:
            continue
        ab = frame.loc["AB"]
        ba = frame.loc["BA"]
        strict_valid = bool(
            ab.selected_snippet_check is not None
            and ba.selected_snippet_check is not None
            and ab.selected_snippet_check == ba.selected_snippet_check
        )
        m_ab = float(ab.margin)
        m_ba = float(ba.margin)
        b = (m_ab + m_ba) / 2.0
        c = (m_ab - m_ba) / 2.0
        boundary = math.isclose(abs(c), abs(b), rel_tol=0.0, abs_tol=1e-12)
        content_tie = math.isclose(c, 0.0, rel_tol=0.0, abs_tol=1e-12)
        relation_valid = abs(c) > abs(b)
        comparable = not boundary and not content_tie
        violation = bool(comparable and strict_valid != relation_valid)
        bc_violations += int(violation)
        boundary_pairs += int(boundary)
        content_ties += int(content_tie)
        details.append(
            {
                "model_key": model_key,
                "pair_id": pair_id,
                "language": ab.language,
                "difficulty": ab.difficulty,
                "strict_valid": strict_valid,
                "m_ab": m_ab,
                "m_ba": m_ba,
                "b": b,
                "c": c,
                "abs_c_gt_abs_b": relation_valid,
                "boundary": boundary,
                "content_tie": content_tie,
                "bc_violation": violation,
            }
        )

    checks = {
        "model_key": model_key,
        "source": str(raw_path.relative_to(ROOT)),
        "source_sha256": sha256(raw_path),
        "calls_checked": len(raw),
        "pairs_checked": raw.pair_id.nunique(),
        "missing_call_keys": len(expected_keys - actual_keys),
        "unexpected_call_keys": len(actual_keys - expected_keys),
        "duplicate_calls": duplicate_calls,
        "parser_mismatches": parser_mismatches,
        "mapping_errors": mapping_errors,
        "missing_logits": missing_logits,
        "margin_orientation_errors": margin_errors,
        "argmax_orientation_errors": argmax_errors,
        "bc_relation_violations_nonboundary_nontie": bc_violations,
        "boundary_pairs": boundary_pairs,
        "content_ties": content_ties,
    }
    checks["pass"] = all(
        checks[key] == 0
        for key in (
            "missing_call_keys",
            "unexpected_call_keys",
            "duplicate_calls",
            "parser_mismatches",
            "mapping_errors",
            "missing_logits",
            "margin_orientation_errors",
            "argmax_orientation_errors",
            "bc_relation_violations_nonboundary_nontie",
        )
    ) and checks["calls_checked"] == 36
    return checks, details


def main() -> int:
    pairs = pd.read_csv(PAIRS)
    sample = fixed_sample(pairs)
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(audit_dir / "SANITY_SAMPLE_18.csv", index=False)

    summaries = []
    details = []
    for model_key, raw_path in RAW_FILES.items():
        summary, model_details = check_model(model_key, raw_path, sample)
        summaries.append(summary)
        details.extend(model_details)
    pd.DataFrame(details).to_csv(audit_dir / "SANITY_BC_DETAILS.csv", index=False)

    report = {
        "audit": "limited_stored_output_sanity_check",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_model_inference_calls": 0,
        "historical_metrics_reproduced": False,
        "pair_manifest": str(PAIRS.relative_to(ROOT)),
        "pair_manifest_sha256": sha256(PAIRS),
        "sampling": "first two protocol_pair_id values per language/difficulty cell",
        "models": summaries,
        "gate_pass": all(item["pass"] for item in summaries),
    }
    (audit_dir / "SANITY_CHECK.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""GA0 integrity gate for the FSE 2027 A-D post-hoc analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
GRID = REPO / "results/grounded_protocol_3lang_20260721"
BATTERY = REPO / "results/rq1_model_battery_3lang_20260723"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_jsonl(
    path: Path,
    *,
    expected_calls: int,
    key_fields: tuple[str, ...],
    expected_orders: set[str],
) -> dict:
    seen: set[tuple[object, ...]] = set()
    duplicate_count = 0
    parse_failures = 0
    argmax_mismatches = 0
    missing_logits = 0
    mapping_failures = 0
    counts: Counter[str] = Counter()
    pair_orders: defaultdict[tuple[object, ...], set[str]] = defaultdict(set)
    models: set[str] = set()
    languages: Counter[str] = Counter()
    conditions: Counter[str] = Counter()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["calls"] += 1
            key = tuple(row.get(field) for field in key_fields)
            if key in seen:
                duplicate_count += 1
            seen.add(key)
            pair_key = tuple(row.get(field) for field in key_fields if field != "order")
            pair_orders[pair_key].add(str(row.get("order")))
            models.add(str(row.get("model")))
            languages[str(row.get("language"))] += 1
            if "condition" in row:
                conditions[str(row.get("condition"))] += 1
            if row.get("parsed_choice") not in {"A", "B"}:
                parse_failures += 1
            if row.get("argmax_matches_parsed") is not True:
                argmax_mismatches += 1
            logits = (row.get("logit_A"), row.get("logit_B"), row.get("margin"))
            if any(value is None or not math.isfinite(float(value)) for value in logits):
                missing_logits += 1
            if "image_mapping_verified" in row and row.get("image_mapping_verified") is not True:
                mapping_failures += 1

    incomplete_pairs = sum(orders != expected_orders for orders in pair_orders.values())
    status = (
        counts["calls"] == expected_calls
        and duplicate_count == 0
        and parse_failures == 0
        and argmax_mismatches == 0
        and missing_logits == 0
        and mapping_failures == 0
        and incomplete_pairs == 0
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "status": "passed" if status else "failed",
        "expected_calls": expected_calls,
        "observed_calls": counts["calls"],
        "unique_keys": len(seen),
        "duplicate_keys": duplicate_count,
        "pair_groups": len(pair_orders),
        "incomplete_pair_groups": incomplete_pairs,
        "parse_failures": parse_failures,
        "verdict_logit_argmax_mismatches": argmax_mismatches,
        "missing_or_nonfinite_logits": missing_logits,
        "image_mapping_failures": mapping_failures,
        "models": sorted(models),
        "language_call_counts": dict(sorted(languages.items())),
        "condition_call_counts": dict(sorted(conditions.items())),
    }


def main() -> None:
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    grid_files = sorted((GRID / "inference/grid/raw").glob("*__full_20260721.jsonl"))
    if len(grid_files) != 2:
        raise RuntimeError(f"Expected two complete grid files, found {len(grid_files)}")
    expected_grid_conditions: set[str] | None = None
    for path in grid_files:
        manifest_path = path.with_suffix(".manifest.json")
        manifest = load_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("completed_calls") != 72000:
            raise RuntimeError(f"Incomplete grid manifest: {manifest_path}")
        conditions = set(manifest["conditions"])
        if len(conditions) != 12:
            raise RuntimeError(f"Grid condition count drift: {manifest_path}")
        if expected_grid_conditions is None:
            expected_grid_conditions = conditions
        elif conditions != expected_grid_conditions:
            raise RuntimeError("Grid condition sets differ across models")
        record = inspect_jsonl(
            path,
            expected_calls=72000,
            key_fields=("condition", "pair_id", "order"),
            expected_orders={"AB", "BA"},
        )
        record.update(
            {
                "run": "grounded_protocol_3lang_20260721",
                "asset": "grid",
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            }
        )
        if set(record["condition_call_counts"]) != expected_grid_conditions:
            record["status"] = "failed"
        if any(count != 6000 for count in record["condition_call_counts"].values()):
            record["status"] = "failed"
        records.append(record)

    battery_pair_ids: set[str] = set()
    with (BATTERY / "data/rq1_pairs_9000.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            battery_pair_ids.add(row["protocol_pair_id"])
    if len(battery_pair_ids) != 9000:
        raise RuntimeError("Battery pair metadata does not contain 9,000 unique protocol IDs")

    model_dirs = ["qwen", "internvl", "gemma", "ministral", "phi"]
    for model_key in model_dirs:
        run_dir = BATTERY / "inference/full" / model_key
        path = run_dir / "raw.jsonl"
        manifest_path = run_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("calls_completed") != 18000:
            raise RuntimeError(f"Incomplete battery manifest: {manifest_path}")
        record = inspect_jsonl(
            path,
            expected_calls=18000,
            key_fields=("pair_id", "order"),
            expected_orders={"AB", "BA"},
        )
        record.update(
            {
                "run": "rq1_model_battery_3lang_20260723",
                "asset": f"battery_{model_key}",
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            }
        )
        records.append(record)

    overall_pass = all(record["status"] == "passed" for record in records)
    report = {
        "gate": "GA0",
        "status": "passed" if overall_pass else "failed",
        "new_model_calls": 0,
        "rules": {
            "grid_files": "Only complete full_20260721 files; pilots and GA0 files excluded",
            "battery_files": "Only five complete inference/full model directories; partial Qwen attempt excluded",
            "checks": [
                "expected call count",
                "unique condition/pair/order or pair/order keys",
                "complete AB/BA allocation",
                "parsed verdict in {A,B}",
                "stored verdict equals captured-logit argmax",
                "finite A/B logits and margin",
                "verified image mapping when field is present",
            ],
        },
        "assets": records,
    }
    (audit_dir / "GA0_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary_fields = [
        "run",
        "asset",
        "status",
        "expected_calls",
        "observed_calls",
        "unique_keys",
        "duplicate_keys",
        "pair_groups",
        "incomplete_pair_groups",
        "parse_failures",
        "verdict_logit_argmax_mismatches",
        "missing_or_nonfinite_logits",
        "image_mapping_failures",
        "path",
        "sha256",
    ]
    with (audit_dir / "GA0_ASSET_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in summary_fields} for row in records)

    print(json.dumps(report, indent=2))
    if not overall_pass:
        raise SystemExit("GA0 failed; downstream analyses are prohibited")


if __name__ == "__main__":
    main()

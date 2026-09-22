#!/usr/bin/env python3
"""Reconstruct and pixel-check every combined pair image."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_combined_pair_images import make_combined  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--render-metadata", required=True)
    parser.add_argument("--combined-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-horizontal-width", type=int, default=1800)
    parser.add_argument("--label-font-size", type=int, default=34)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    pairs = pd.read_csv(repo / args.pairs)
    renders = pd.read_csv(repo / args.render_metadata).set_index("rq0_id")
    combined = pd.read_csv(repo / args.combined_metadata)
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    expected_keys = {(str(pair_id), order) for pair_id in pairs["pair_id"] for order in ("AB", "BA")}
    actual_keys = set(zip(combined["pair_id"].astype(str), combined["order"].astype(str)))
    failures: list[dict[str, str]] = []
    if expected_keys != actual_keys or len(combined) != len(expected_keys):
        failures.append(
            {
                "pair_id": "__metadata__",
                "order": "",
                "error": f"key mismatch missing={len(expected_keys - actual_keys)} extra={len(actual_keys - expected_keys)} rows={len(combined)}",
            }
        )

    pair_lookup = pairs.set_index("pair_id")
    orientation_counts = {"horizontal": 0, "vertical": 0}
    checked = 0
    for row in combined.itertuples(index=False):
        try:
            pair = pair_lookup.loc[row.pair_id]
            expected_a, expected_b = (
                (pair.snippet_i, pair.snippet_j) if row.order == "AB" else (pair.snippet_j, pair.snippet_i)
            )
            if row.snippet_a != expected_a or row.snippet_b != expected_b:
                raise RuntimeError(f"panel mapping mismatch expected={expected_a},{expected_b}")
            with Image.open(renders.loc[expected_a, "image_path"]) as raw_a, Image.open(
                renders.loc[expected_b, "image_path"]
            ) as raw_b:
                image_a = raw_a.convert("RGB")
                image_b = raw_b.convert("RGB")
                horizontal_width = args.padding * 3 + image_a.width + image_b.width
                orientation = "horizontal" if horizontal_width <= args.max_horizontal_width else "vertical"
                expected = make_combined(image_a, image_b, orientation, args.label_font_size, args.padding)
            with Image.open(row.image_path) as raw_actual:
                actual = raw_actual.convert("RGB")
            if row.orientation != orientation:
                raise RuntimeError(f"orientation mismatch metadata={row.orientation} expected={orientation}")
            if actual.size != expected.size or actual.size != (int(row.width), int(row.height)):
                raise RuntimeError(f"dimension mismatch actual={actual.size} expected={expected.size}")
            if ImageChops.difference(actual, expected).getbbox() is not None:
                raise RuntimeError("pixel mismatch against deterministic reconstruction")
            orientation_counts[orientation] += 1
            checked += 1
        except Exception as exc:
            failures.append({"pair_id": str(row.pair_id), "order": str(row.order), "error": f"{type(exc).__name__}: {exc}"})
        if (checked + len(failures)) % 1000 == 0:
            print(f"audited {checked + len(failures)}/{len(combined)} failures={len(failures)}", flush=True)

    pd.DataFrame(failures, columns=["pair_id", "order", "error"]).to_csv(output.with_suffix(".failures.csv"), index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures and checked == len(combined) else "failed",
        "rows_expected": int(2 * len(pairs)),
        "rows_checked": checked,
        "failures": len(failures),
        "orientation_counts": orientation_counts,
        "method": "full pixel equality against deterministic reconstruction from audited individual renders",
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

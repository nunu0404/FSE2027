#!/usr/bin/env python3
"""Freeze the seed-42 robustness subset and existing-style combined canvases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
PAIR_PATH = ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
SOURCE_PATH = ROOT / "results/grounded_protocol_3lang_20260721/data/render_snippets.csv"
LEGACY_COMBINER = ROOT / "experiments/rq0_viability/scripts/render_combined_pair_images.py"
COMBINED_DIR = Path(
    "/ANON/scratch_rq1/latest_vlm_extension_20260830/robustness_combined"
)
CONFIG = {
    "max_horizontal_width": 1800,
    "label_font_size": 34,
    "padding": 18,
    "layout_rule": "horizontal if total width <= max_horizontal_width else vertical",
    "label_text": ["Code A", "Code B"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_combiner():
    spec = importlib.util.spec_from_file_location("frozen_combiner", LEGACY_COMBINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(LEGACY_COMBINER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    selected = []
    for (language, difficulty), group in pairs.groupby(
        ["language", "difficulty"], sort=True
    ):
        selected.append(group.sample(n=100, random_state=42))
    subset = pd.concat(selected, ignore_index=True).sort_values(
        ["language", "difficulty", "protocol_pair_id"]
    )
    if len(subset) != 900 or subset.protocol_pair_id.nunique() != 900:
        raise RuntimeError("robustness subset allocation drift")
    if not (subset.groupby(["language", "difficulty"]).size() == 100).all():
        raise RuntimeError("robustness cell allocation drift")
    source = pd.read_csv(SOURCE_PATH).set_index("rq0_id")
    ids = set(subset.snippet_i) | set(subset.snippet_j)
    if ids - set(source.index):
        raise RuntimeError("source text is missing for selected snippets")
    subset["source_i"] = subset.snippet_i.map(source.raw_code)
    subset["source_j"] = subset.snippet_j.map(source.raw_code)

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    combiner = load_legacy_combiner()
    rows = []
    for index, pair in enumerate(subset.itertuples(index=False), 1):
        for order, first_path, second_path, first_id, second_id in (
            (
                "AB",
                pair.image_i_path,
                pair.image_j_path,
                pair.snippet_i,
                pair.snippet_j,
            ),
            (
                "BA",
                pair.image_j_path,
                pair.image_i_path,
                pair.snippet_j,
                pair.snippet_i,
            ),
        ):
            with Image.open(ROOT / first_path) as raw_a, Image.open(
                ROOT / second_path
            ) as raw_b:
                image_a = raw_a.convert("RGB")
                image_b = raw_b.convert("RGB")
                horizontal_width = CONFIG["padding"] * 3 + image_a.width + image_b.width
                orientation = (
                    "horizontal"
                    if horizontal_width <= CONFIG["max_horizontal_width"]
                    else "vertical"
                )
                combined = combiner.make_combined(
                    image_a,
                    image_b,
                    orientation,
                    CONFIG["label_font_size"],
                    CONFIG["padding"],
                )
            output_path = COMBINED_DIR / f"{pair.protocol_pair_id}__{order}.png"
            combined.save(output_path)
            rows.append(
                {
                    "pair_id": pair.protocol_pair_id,
                    "order": order,
                    "snippet_first": first_id,
                    "snippet_second": second_id,
                    "image_path": str(output_path),
                    "image_sha256": sha256(output_path),
                    "width": combined.width,
                    "height": combined.height,
                    "orientation": orientation,
                }
            )
        if index % 100 == 0:
            print(f"ROBUSTNESS_RENDER {index}/900", flush=True)

    metadata = pd.DataFrame(rows)
    if len(metadata) != 1800 or metadata[["pair_id", "order"]].duplicated().any():
        raise RuntimeError("combined metadata completeness failure")
    for path in metadata.image_path:
        with Image.open(path) as image:
            image.verify()
    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(data_dir / "robustness_pairs_900_seed42.csv", index=False)
    metadata.to_csv(data_dir / "robustness_combined_metadata.csv", index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "pandas group.sample(n=100, random_state=42) per language/difficulty",
        "pairs": len(subset),
        "calls_per_model": len(subset) * 4 * 2,
        "language_difficulty": {
            f"{language}|{difficulty}": int(count)
            for (language, difficulty), count in subset.groupby(
                ["language", "difficulty"]
            ).size().items()
        },
        "combined_images": len(metadata),
        "combined_layouts": metadata.orientation.value_counts().to_dict(),
        "pair_source_sha256": sha256(PAIR_PATH),
        "source_text_sha256": sha256(SOURCE_PATH),
        "legacy_combiner": str(LEGACY_COMBINER.relative_to(ROOT)),
        "legacy_combiner_sha256": sha256(LEGACY_COMBINER),
        "combiner_config": CONFIG,
        "combined_output_dir": str(COMBINED_DIR),
        "images_verified": True,
        "gate_pass": True,
    }
    (data_dir / "ROBUSTNESS_ASSET_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

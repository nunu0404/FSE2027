#!/usr/bin/env python3
"""Prepare immutable data copies and corrected A/B rendering artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
OLD = ROOT / "results/protocol_unified_3lang_20260720"
sys.path.insert(0, str(OUT / "code"))
from token_preserving_renderer import normalize_code, render_code  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_stats(path: Path) -> dict[str, float | int | bool]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = array.mean(axis=2)
    return {
        "image_width": int(array.shape[1]),
        "image_height": int(array.shape[0]),
        "pixel_std": float(array.std()),
        "gradient_energy": float(np.mean(np.diff(gray, axis=0) ** 2) + np.mean(np.diff(gray, axis=1) ** 2)),
        "nonblank": bool(array.std() > 0),
    }


def conditions(config: dict) -> list[dict]:
    render = config["rendering"]
    return [
        {
            "condition": f"{mode}__fs{font_size}__wrap{wrap_column}__lnon",
            "display_mode": mode,
            "style": style,
            "font_size": font_size,
            "wrap_column": wrap_column,
            "line_numbers": True,
        }
        for mode, style in render["display_modes"].items()
        for font_size in render["font_sizes"]
        for wrap_column in render["wrap_columns"]
    ]


def prepare_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(OLD / "data/ab_pairs_seed42.csv")
    snippets = pd.read_csv(OLD / "data/render_snippets.csv")
    if pairs.groupby("language").size().to_dict() != {"cuda": 1000, "java": 1000, "python": 1000}:
        raise ValueError("expected exactly 1,000 fixed pairs per language")
    pairs["difficulty_legacy"] = pairs["difficulty"]
    pairs["difficulty_rank"] = ""
    for language, indices in pairs.groupby("language").groups.items():
        ordered = pairs.loc[indices].sort_values(["abs_z_diff", "pair_id"]).index
        labels = np.array(["near"] * 334 + ["middle"] * 333 + ["far"] * 333)
        pairs.loc[ordered, "difficulty_rank"] = labels
    pairs["difficulty"] = pairs["difficulty_rank"]
    pairs.to_csv(data_dir / "pairs_seed42_grounded.csv", index=False)
    snippets.to_csv(data_dir / "render_snippets.csv", index=False)
    pd.DataFrame(conditions(config)).to_csv(data_dir / "render_conditions.csv", index=False)
    return pairs, snippets


def render_grid(config: dict, snippets: pd.DataFrame) -> pd.DataFrame:
    render = config["rendering"]
    font_path = ROOT / render["font_path"]
    records = []
    total = len(snippets) * 12
    done = 0
    for condition in conditions(config):
        for row in snippets.itertuples(index=False):
            output = OUT / "rendered/grid" / condition["condition"] / f"{row.rq0_id}.png"
            layout = render_code(
                row.raw_code, row.language, output, font_path,
                condition["style"], condition["font_size"], condition["wrap_column"],
                line_numbers=True, tab_width=render["tab_width"], image_pad=render["image_pad"],
                line_gap=render["line_gap"],
            )
            records.append({
                "rq0_id": row.rq0_id,
                "language": row.language,
                **condition,
                **layout,
                **image_stats(output),
                "source_sha256": hashlib.sha256(normalize_code(row.raw_code).encode()).hexdigest(),
                "image_sha256": sha256_file(output),
                "image_path": str(output.relative_to(ROOT)),
            })
            done += 1
            if done % 500 == 0 or done == total:
                print(f"grid {done}/{total}", flush=True)
    metadata = pd.DataFrame(records)
    path = OUT / "rendered/metadata/grid_render_metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(path, index=False)
    return metadata


def render_b(config: dict, snippets: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    render = config["rendering"]
    font_path = ROOT / render["font_path"]
    baseline_condition = render["baseline_condition"]
    baseline = grid[grid.condition.eq(baseline_condition)].set_index("rq0_id")
    sigmas = {"gaussian_sigma_1": 1.0, "gaussian_sigma_2": 2.0, "gaussian_sigma_4": 4.0}
    records = []
    for index, row in enumerate(snippets.itertuples(index=False), 1):
        paths = {name: OUT / "rendered/perturbation" / name / f"{row.rq0_id}.png" for name in config["perturbations"]["conditions"]}
        paths["baseline"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / baseline.loc[row.rq0_id, "image_path"], paths["baseline"])
        for condition, kwargs in {
            "no_indent": {"remove_indent": True},
            "no_blank_lines": {"remove_blank_lines": True},
        }.items():
            render_code(
                row.raw_code, row.language, paths[condition], font_path, "monokai", 20, 80,
                line_numbers=True, tab_width=render["tab_width"], image_pad=render["image_pad"],
                line_gap=render["line_gap"], **kwargs,
            )
        with Image.open(paths["baseline"]) as source:
            rgb = source.convert("RGB")
            for condition, sigma in sigmas.items():
                paths[condition].parent.mkdir(parents=True, exist_ok=True)
                rgb.filter(ImageFilter.GaussianBlur(radius=sigma)).save(paths[condition])
        original_lines = normalize_code(row.raw_code).split("\n")
        has_indent = any(line.startswith((" ", "\t")) for line in original_lines)
        has_blank = any(not line.strip() for line in original_lines)
        for condition, path in paths.items():
            records.append({
                "rq0_id": row.rq0_id,
                "language": row.language,
                "condition": condition,
                "sigma_pixels": sigmas.get(condition, 0.0),
                "cue_present": has_indent if condition == "no_indent" else has_blank if condition == "no_blank_lines" else True,
                **image_stats(path),
                "source_sha256": hashlib.sha256(normalize_code(row.raw_code).encode()).hexdigest(),
                "image_sha256": sha256_file(path),
                "image_path": str(path.relative_to(ROOT)),
            })
        if index % 50 == 0 or index == len(snippets):
            print(f"perturbation {index}/{len(snippets)}", flush=True)
    metadata = pd.DataFrame(records)
    path = OUT / "rendered/metadata/perturbation_render_metadata.csv"
    metadata.to_csv(path, index=False)
    return metadata


def main() -> int:
    config = json.loads((OUT / "config/protocol_grounded.json").read_text())
    _, snippets = prepare_data(config)
    grid = render_grid(config, snippets)
    perturbation = render_b(config, snippets, grid)
    manifest = {
        "protocol_version": config["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snippets": len(snippets),
        "grid_conditions": grid.condition.nunique(),
        "grid_images": len(grid),
        "perturbation_conditions": perturbation.condition.nunique(),
        "perturbation_images": len(perturbation),
        "renderer": "token-preserving lex-first soft wrap",
        "source_protocol_untouched": True,
    }
    (OUT / "rendered/RENDER_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

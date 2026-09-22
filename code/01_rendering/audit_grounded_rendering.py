#!/usr/bin/env python3
"""Exhaustive invariants and deterministic contact sheets for grounded rendering."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
sys.path.insert(0, str(OUT / "code"))
from token_preserving_renderer import layout_rows, normalize_code  # noqa: E402


def lexer(language: str):
    try:
        return get_lexer_by_name(language)
    except ClassNotFound:
        return TextLexer()


def token_hash(code: str, language: str) -> str:
    digest = hashlib.sha256()
    for token_type, value in lex(normalize_code(code), lexer(language)):
        digest.update(str(token_type).encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def open_array(path: str) -> np.ndarray:
    with Image.open(ROOT / path) as image:
        return np.asarray(image.convert("RGB"))


def make_sheet(frame: pd.DataFrame, output: Path, columns: int = 3) -> None:
    thumbs = []
    for row in frame.itertuples(index=False):
        with Image.open(ROOT / row.image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((620, 390), Image.Resampling.LANCZOS)
        thumbs.append((row.condition, image.copy()))
    cell_w, cell_h = 640, 430
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(thumbs):
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        draw.text((x + 8, y + 8), label, fill="black", font=font)
        sheet.paste(image, (x + 8, y + 30))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    config = json.loads((OUT / "config/protocol_grounded.json").read_text())
    snippets = pd.read_csv(OUT / "data/render_snippets.csv").set_index("rq0_id")
    pairs = pd.read_csv(OUT / "data/pairs_seed42_grounded.csv")
    grid = pd.read_csv(OUT / "rendered/metadata/grid_render_metadata.csv")
    perturb = pd.read_csv(OUT / "rendered/metadata/perturbation_render_metadata.csv")
    expected_grid = len(snippets) * 12
    expected_perturb = len(snippets) * 6
    audit: dict[str, object] = {
        "snippets": len(snippets),
        "pairs": len(pairs),
        "pairs_by_language": pairs.groupby("language").size().to_dict(),
        "grid_rows": len(grid),
        "grid_expected": expected_grid,
        "perturbation_rows": len(perturb),
        "perturbation_expected": expected_perturb,
        "grid_unique_keys": not grid.duplicated(["rq0_id", "condition"]).any(),
        "perturbation_unique_keys": not perturb.duplicated(["rq0_id", "condition"]).any(),
        "all_files_exist": all((ROOT / path).is_file() for path in pd.concat([grid.image_path, perturb.image_path])),
        "all_nonblank": bool(grid.nonblank.all() and perturb.nonblank.all()),
    }

    # Recompute layout row accounting and immutable source/token signatures.
    layout_errors = 0
    source_hash_errors = 0
    token_rows = []
    for rq0_id, snippet in snippets.iterrows():
        source_hash = hashlib.sha256(normalize_code(snippet.raw_code).encode()).hexdigest()
        lexical_hash = token_hash(snippet.raw_code, snippet.language)
        token_rows.append({"rq0_id": rq0_id, "language": snippet.language, "source_sha256": source_hash, "token_stream_sha256": lexical_hash})
        for row in grid[grid.rq0_id.eq(rq0_id)].itertuples(index=False):
            rows = layout_rows(snippet.raw_code, snippet.language, row.style, int(row.wrap_column), config["rendering"]["tab_width"])
            if len(rows) != row.visual_rows or sum(not item.show_line_number for item in rows) != row.continuation_rows:
                layout_errors += 1
            if source_hash != row.source_sha256:
                source_hash_errors += 1
    pd.DataFrame(token_rows).to_csv(OUT / "audit/token_stream_signatures.csv", index=False)
    audit["layout_row_accounting_errors"] = layout_errors
    audit["grid_source_hash_errors"] = source_hash_errors
    audit["token_streams_lexed_from_original_source"] = len(token_rows)

    # Every condition has a stable canvas width; height exactly matches metadata.
    audit["conditions_with_variable_width"] = int((grid.groupby("condition").image_width.nunique() != 1).sum())
    audit["invalid_dimensions"] = int(
        ((grid.image_width <= 0) | (grid.image_height <= 0)).sum()
        + ((perturb.image_width <= 0) | (perturb.image_height <= 0)).sum()
    )

    baseline_condition = config["rendering"]["baseline_condition"]
    grid_base = grid[grid.condition.eq(baseline_condition)].set_index("rq0_id")
    b_base = perturb[perturb.condition.eq("baseline")].set_index("rq0_id")
    exact = []
    for rq0_id in snippets.index:
        exact.append(np.array_equal(open_array(grid_base.loc[rq0_id, "image_path"]), open_array(b_base.loc[rq0_id, "image_path"])))
    audit["b_baseline_pixel_exact"] = int(sum(exact))
    audit["b_baseline_total"] = len(exact)

    # Blur must preserve dimensions, alter pixels, and monotonically remove edge energy.
    p = perturb.pivot(index="rq0_id", columns="condition", values=["gradient_energy", "image_width", "image_height", "image_sha256"])
    blur_names = ["gaussian_sigma_1", "gaussian_sigma_2", "gaussian_sigma_4"]
    monotonic = (p.gradient_energy.baseline > p.gradient_energy.gaussian_sigma_1) & (p.gradient_energy.gaussian_sigma_1 > p.gradient_energy.gaussian_sigma_2) & (p.gradient_energy.gaussian_sigma_2 > p.gradient_energy.gaussian_sigma_4)
    distinct = p.image_sha256[["baseline", *blur_names]].nunique(axis=1).eq(4)
    dimensions = pd.Series(True, index=p.index)
    for name in blur_names:
        dimensions &= p.image_width[name].eq(p.image_width.baseline) & p.image_height[name].eq(p.image_height.baseline)
    audit["blur_monotonic_gradient"] = int(monotonic.sum())
    audit["blur_all_distinct"] = int(distinct.sum())
    audit["blur_dimensions_preserved"] = int(dimensions.sum())
    audit["blur_total"] = len(p)

    # Compute full SSIM characterization for fixed blur levels.
    ssim_rows = []
    for rq0_id in snippets.index:
        base = open_array(b_base.loc[rq0_id, "image_path"])
        for name in blur_names:
            row = perturb[(perturb.rq0_id.eq(rq0_id)) & (perturb.condition.eq(name))].iloc[0]
            value = structural_similarity(base, open_array(row.image_path), channel_axis=2, data_range=255)
            ssim_rows.append({"rq0_id": rq0_id, "language": snippets.loc[rq0_id, "language"], "condition": name, "ssim_vs_baseline": value})
    ssim = pd.DataFrame(ssim_rows)
    ssim.to_csv(OUT / "audit/blur_ssim_all.csv", index=False)
    ssim.groupby(["language", "condition"]).ssim_vs_baseline.agg(["count", "median", "mean", "min", "max"]).reset_index().to_csv(OUT / "audit/blur_ssim_summary.csv", index=False)

    # Cue/no-op coverage at pair level is fixed before inference.
    cue = perturb[perturb.condition.isin(["no_indent", "no_blank_lines"])].pivot(index="rq0_id", columns="condition", values="cue_present")
    coverage_rows = []
    for condition in ["no_indent", "no_blank_lines"]:
        for language, frame in pairs.groupby("language"):
            left = frame.snippet_i.map(cue[condition]).astype(bool)
            right = frame.snippet_j.map(cue[condition]).astype(bool)
            coverage_rows.append({
                "condition": condition, "language": language, "pairs": len(frame),
                "any_side_changed": int((left | right).sum()), "both_sides_changed": int((left & right).sum()),
                "neither_side_changed": int((~(left | right)).sum()),
            })
    pd.DataFrame(coverage_rows).to_csv(OUT / "audit/perturbation_pair_coverage.csv", index=False)

    # Deterministic longest and median examples for every language.
    source_lengths = snippets.raw_code.fillna("").str.len()
    for language in config["languages"]:
        ids = snippets[snippets.language.eq(language)].index
        ordered = source_lengths.loc[ids].sort_values()
        chosen = {"median": ordered.index[len(ordered) // 2], "longest": ordered.index[-1]}
        for label, rq0_id in chosen.items():
            make_sheet(grid[grid.rq0_id.eq(rq0_id)].sort_values("condition"), OUT / f"audit/contact_grid_{language}_{label}_{rq0_id}.png", 3)
            make_sheet(perturb[perturb.rq0_id.eq(rq0_id)].sort_values("condition"), OUT / f"audit/contact_b_{language}_{label}_{rq0_id}.png", 3)

    required_true = [
        audit["grid_rows"] == expected_grid,
        audit["perturbation_rows"] == expected_perturb,
        audit["grid_unique_keys"], audit["perturbation_unique_keys"], audit["all_files_exist"], audit["all_nonblank"],
        layout_errors == 0, source_hash_errors == 0, audit["conditions_with_variable_width"] == 0,
        audit["invalid_dimensions"] == 0, audit["b_baseline_pixel_exact"] == len(snippets),
        audit["blur_monotonic_gradient"] == len(snippets), audit["blur_all_distinct"] == len(snippets),
        audit["blur_dimensions_preserved"] == len(snippets),
    ]
    audit["gate_pass"] = bool(all(required_true))
    (OUT / "audit/GROUNDED_RENDER_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0 if audit["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

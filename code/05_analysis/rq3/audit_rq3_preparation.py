#!/usr/bin/env python3
"""Independent completeness, provenance, and visual-layout audit for RQ3 assets."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
RQ3 = OUT / "rq3"
sys.path.insert(0, str(OUT / "vendor"))

from tree_sitter import Language, Parser  # noqa: E402
import tree_sitter_cpp  # noqa: E402
import tree_sitter_java  # noqa: E402
import tree_sitter_python  # noqa: E402


ID_TYPES = {"identifier", "field_identifier", "type_identifier", "namespace_identifier"}
PARSERS = {
    "java": Parser(Language(tree_sitter_java.language())),
    "python": Parser(Language(tree_sitter_python.language())),
    "cuda": Parser(Language(tree_sitter_cpp.language())),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def identifiers(language: str, code: str) -> set[str]:
    data = code.encode("utf-8")
    root = PARSERS[language].parse(data).root_node
    return {
        data[node.start_byte:node.end_byte].decode("utf-8")
        for node in nodes(root)
        if node.type in ID_TYPES and not node.children
    }


def nonopaque_comments(language: str, code: str) -> list[str]:
    comments = []
    for token_type, value in lex(code, get_lexer_by_name(language)):
        if token_type not in Comment or token_type in Comment.Preproc:
            continue
        if value.strip() and "opaque semantic cue" not in value:
            comments.append(value.strip())
    return comments


def contacts(base: pd.DataFrame, metadata: pd.DataFrame) -> list[str]:
    lookup = metadata.set_index(["base_id", "variant"]).image_path.to_dict()
    variants = ["golden", "ugly_gold", "beautiful_trash", "ugly_trash"]
    outputs = []
    font = ImageFont.load_default()
    for language, frame in base.groupby("language", sort=True):
        chosen = []
        for stratum in ["low", "mid", "high"]:
            group = frame[frame.score_stratum.eq(stratum)].sort_values("base_id")
            chosen.append(group.iloc[len(group) // 2])
        thumbs = []
        for row in chosen:
            line = []
            for variant in variants:
                with Image.open(ROOT / lookup[(row.base_id, variant)]) as image:
                    item = image.convert("RGB")
                    item.thumbnail((520, 430))
                    tile = Image.new("RGB", (540, 470), "white")
                    tile.paste(item, ((540 - item.width) // 2, 28))
                    draw = ImageDraw.Draw(tile)
                    draw.text((8, 8), f"{row.base_id} | {row.score_stratum} | {variant}", fill="black", font=font)
                    line.append(tile)
            thumbs.append(line)
        sheet = Image.new("RGB", (540 * 4, 470 * 3), "#cccccc")
        for y, line in enumerate(thumbs):
            for x, tile in enumerate(line):
                sheet.paste(tile, (x * 540, y * 470))
        target = RQ3 / "audit" / f"contact_rq3_{language}.png"
        sheet.save(target)
        outputs.append(str(target.relative_to(ROOT)))
    return outputs


def main() -> int:
    base = pd.read_csv(RQ3 / "data/rq3_grounded_bases_seed42.csv")
    sources = pd.read_csv(RQ3 / "data/rq3_variant_sources.csv")
    mutations = pd.read_csv(RQ3 / "data/rq3_mutation_log.csv")
    metadata = pd.read_csv(RQ3 / "data/rq3_render_metadata.csv")
    contrasts = pd.read_csv(RQ3 / "data/rq3_contrast_manifest.csv")
    review = pd.read_csv(RQ3 / "review/GC1_REVIEW_FORM.csv")

    missing_images, hash_errors, dimension_errors, blank_images = [], [], [], []
    for row in metadata.itertuples(index=False):
        path = ROOT / row.image_path
        if not path.is_file():
            missing_images.append(row.image_path)
            continue
        if sha256(path) != row.image_sha256:
            hash_errors.append(row.image_path)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            if image.width != row.image_width or image.height != row.image_height:
                dimension_errors.append(row.image_path)
            if float(pixels.std()) <= 0.5:
                blank_images.append(row.image_path)
        expected_height = (20 if row.visual_quality == "clean" else 16) + row.visual_rows * row.row_height
        if row.image_height != expected_height:
            dimension_errors.append(row.image_path + ":row_height")

    source_wide = metadata.pivot(index="base_id", columns="variant", values="source_sha256")
    pair_source_mismatches = int((
        ~source_wide.golden.eq(source_wide.ugly_gold)
        | ~source_wide.beautiful_trash.eq(source_wide.ugly_trash)
    ).sum())
    per_base_families = mutations.groupby("base_id").family.nunique()
    per_base_mutations = mutations.groupby("base_id").size()
    source_pairs = sources.pivot(index="base_id", columns="semantic_integrity", values="code")
    language_by_base = base.set_index("base_id").language.to_dict()
    identifier_residuals = {}
    comment_residuals = {}
    for base_id, row in source_pairs.iterrows():
        language = language_by_base[base_id]
        overlap = sorted(identifiers(language, row.gold) & identifiers(language, row.trash))
        overlap = [value for value in overlap if value and not value.startswith("opaque_")]
        if overlap:
            identifier_residuals[base_id] = overlap
        remaining_comments = nonopaque_comments(language, row.trash)
        if remaining_comments:
            comment_residuals[base_id] = remaining_comments
    contrast_pairs = contrasts.assign(pair=contrasts.apply(lambda r: tuple(sorted((r.variant_i, r.variant_j))), axis=1))
    expected_pairs = {
        ("golden", "ugly_gold"), ("beautiful_trash", "golden"), ("golden", "ugly_trash"),
        ("beautiful_trash", "ugly_gold"), ("ugly_gold", "ugly_trash"),
        ("beautiful_trash", "ugly_trash"),
    }
    contrast_complete = contrast_pairs.groupby("base_id").pair.apply(lambda values: set(values) == expected_pairs)
    contacts_created = contacts(base, metadata)
    result = {
        "bases": len(base), "bases_by_language": base.groupby("language").size().to_dict(),
        "strata": {f"{a}/{b}": int(v) for (a, b), v in base.groupby(["language", "score_stratum"]).size().items()},
        "sources": len(sources), "images": len(metadata), "contrasts": len(contrasts),
        "unique_base_variant_keys": not metadata.duplicated(["base_id", "variant"]).any(),
        "missing_images": len(missing_images), "image_hash_errors": len(hash_errors),
        "dimension_or_row_height_errors": len(set(dimension_errors)), "blank_images": len(blank_images),
        "pair_source_hash_mismatches": pair_source_mismatches,
        "original_identifier_cue_residual_bases": len(identifier_residuals),
        "nonopaque_comment_residual_bases": len(comment_residuals),
        "parse_error_increases": int((
            sources.pivot(index="base_id", columns="semantic_integrity", values="parse_error_nodes").trash
            > sources.pivot(index="base_id", columns="semantic_integrity", values="parse_error_nodes").gold
        ).sum()),
        "minimum_mutation_families": int(per_base_families.min()),
        "mutation_count_min": int(per_base_mutations.min()),
        "mutation_count_median": float(per_base_mutations.median()),
        "mutation_count_max": int(per_base_mutations.max()),
        "contrast_complete_bases": int(contrast_complete.sum()),
        "review_items": len(review), "review_by_language": review.groupby("language").size().to_dict(),
        "contact_sheets": contacts_created,
        "author_semantic_gate": "pending",
    }
    result["automatic_gate_pass"] = bool(
        result["bases"] == 300 and result["sources"] == 600 and result["images"] == 1200
        and result["contrasts"] == 1800 and result["unique_base_variant_keys"]
        and result["missing_images"] == result["image_hash_errors"] == 0
        and result["dimension_or_row_height_errors"] == result["blank_images"] == 0
        and result["pair_source_hash_mismatches"] == result["parse_error_increases"] == 0
        and result["original_identifier_cue_residual_bases"] == 0
        and result["nonopaque_comment_residual_bases"] == 0
        and result["minimum_mutation_families"] >= 2 and result["contrast_complete_bases"] == 300
        and result["review_items"] == 60
    )
    (RQ3 / "audit/RQ3_AUTOMATIC_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

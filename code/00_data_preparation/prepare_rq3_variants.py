#!/usr/bin/env python3
"""Prepare deterministic, parser-audited RQ3 semantic x visual variants."""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
sys.path.insert(0, str(OUT / "vendor"))
sys.path.insert(0, str(OUT / "code"))

from tree_sitter import Language, Parser  # noqa: E402
import tree_sitter_cpp  # noqa: E402
import tree_sitter_java  # noqa: E402
import tree_sitter_python  # noqa: E402
from token_preserving_renderer import normalize_code, render_code  # noqa: E402


SEED = 42
MUTATION_INELIGIBLE = {"dorn_python_016"}  # Fragmented multiline literals cannot retain its baseline parse under identifier mutation.
ID_TYPES = {"identifier", "field_identifier", "type_identifier", "namespace_identifier"}
BEHAVIOR_TYPES = {
    "java": {"method_invocation", "assignment_expression", "if_statement", "for_statement",
             "enhanced_for_statement", "while_statement", "return_statement", "binary_expression",
             "object_creation_expression"},
    "python": {"call", "assignment", "augmented_assignment", "if_statement", "for_statement",
               "while_statement", "return_statement", "binary_operator", "comparison_operator"},
    "cuda": {"call_expression", "assignment_expression", "if_statement", "for_statement",
             "while_statement", "return_statement", "binary_expression", "update_expression"},
}
STRING_TYPES = {
    "string", "concatenated_string", "string_literal", "character_literal", "char_literal", "raw_string_literal"
}
NUMBER_TYPES = {
    "integer", "float", "decimal_integer_literal", "decimal_floating_point_literal",
    "hex_integer_literal", "binary_integer_literal", "octal_integer_literal", "number_literal"
}
OPERATOR_MAP = {
    "+": "-", "-": "+", "*": "%", "/": "*", "%": "+",
    "<": ">=", ">": "<=", "<=": ">", ">=": "<", "==": "!=", "!=": "==",
    "&&": "||", "||": "&&", "and": "or", "or": "and", "&": "|", "|": "^", "^": "&",
    "<<": ">>", ">>": "<<",
}
OPERATOR_PARENT_TYPES = {
    "binary_expression", "binary_operator", "comparison_operator", "boolean_operator"
}
PARSERS = {
    "java": Parser(Language(tree_sitter_java.language())),
    "python": Parser(Language(tree_sitter_python.language())),
    "cuda": Parser(Language(tree_sitter_cpp.language())),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def parse_stats(language: str, code: str) -> dict[str, int | bool]:
    data = code.encode("utf-8")
    root = PARSERS[language].parse(data).root_node
    all_nodes = list(nodes(root))
    return {
        "has_error": bool(root.has_error),
        "error_nodes": sum(node.type == "ERROR" or node.is_missing for node in all_nodes),
        "identifier_nodes": sum(node.type in ID_TYPES for node in all_nodes),
        "behavior_nodes": sum(node.type in BEHAVIOR_TYPES[language] for node in all_nodes),
    }


def select_bases(snippets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for row in snippets.itertuples(index=False):
        stats = parse_stats(row.language, normalize_code(row.raw_code))
        records.append({"rq0_id": row.rq0_id, "language": row.language, **stats})
    eligibility = pd.DataFrame(records)
    merged = snippets.merge(eligibility, on=["rq0_id", "language"], validate="one_to_one")
    merged["eligible"] = (
        merged.identifier_nodes.ge(3) & merged.behavior_nodes.ge(2)
        & ~merged.rq0_id.isin(MUTATION_INELIGIBLE)
    )
    feasible = []
    for row in merged.itertuples(index=False):
        if not row.eligible:
            feasible.append(False)
            continue
        try:
            mutate(row.language, normalize_code(row.raw_code), f"candidate_{row.rq0_id}")
            feasible.append(True)
        except ValueError:
            feasible.append(False)
    merged["mutation_feasible"] = feasible
    merged["eligible"] &= merged.mutation_feasible
    selected = []
    for language, frame in merged[merged.eligible].groupby("language", sort=True):
        ordered = frame.sort_values(["score_for_stratum", "rq0_id"]).reset_index(drop=True)
        ordered["score_stratum"] = pd.qcut(ordered.index.to_series().rank(method="first"), 3,
                                            labels=["low", "mid", "high"])
        rng = random.Random(SEED + {"java": 1, "python": 2, "cuda": 3}[language])
        for stratum, count in (("low", 34), ("mid", 33), ("high", 33)):
            pool = ordered[ordered.score_stratum.eq(stratum)].index.tolist()
            if len(pool) < count:
                raise ValueError(f"not enough eligible {language}/{stratum}: {len(pool)} < {count}")
            chosen = sorted(rng.sample(pool, count))
            selected.append(ordered.loc[chosen])
    result = pd.concat(selected, ignore_index=True).sort_values(["language", "score_stratum", "score_for_stratum", "rq0_id"])
    result["base_index"] = result.groupby("language").cumcount()
    result["base_id"] = result.apply(lambda row: f"rq3g_{row.language}_{int(row.base_index):03d}", axis=1)
    if result.groupby("language").size().to_dict() != {"cuda": 100, "java": 100, "python": 100}:
        raise ValueError("base allocation failed")
    return result, merged


def replacement_for_string(node_type: str, token: str, serial: int) -> str:
    if node_type in {"character_literal", "char_literal"}:
        return "'z'"
    return f'"opaque_{serial:03d}"'


def scrub_lexical_comments(language: str, code: str) -> tuple[str, int]:
    parts, changed = [], 0
    for token_type, value in lex(code, get_lexer_by_name(language)):
        if token_type not in Comment or token_type in Comment.Preproc or "opaque semantic cue" in value:
            parts.append(value)
            continue
        line_count = value.count("\n") + 1
        if value.lstrip().startswith("/*"):
            replacement = "/*" + ("\n" * (line_count - 1)) + " opaque semantic cue */"
        else:
            marker = "#" if language == "python" else "//"
            replacement = "\n".join(
                f"{marker} opaque semantic cue" + (" \\" if line.rstrip().endswith("\\") else "")
                for line in value.split("\n")
            )
        parts.append(replacement)
        changed += replacement != value
    return "".join(parts), changed


def mutate(language: str, code: str, base_id: str) -> tuple[str, list[dict[str, object]], list[str]]:
    source = normalize_code(code)
    data = source.encode("utf-8")
    root = PARSERS[language].parse(data).root_node
    identifiers = [node for node in nodes(root) if node.type in ID_TYPES and not node.children]
    if len(identifiers) < 3:
        raise ValueError(f"insufficient identifiers after selection: {base_id}")
    original_names = [data[node.start_byte:node.end_byte].decode("utf-8") for node in identifiers]
    unique_names = list(dict.fromkeys(original_names))
    salt = hashlib.sha256(base_id.encode()).hexdigest()[:6]
    masked_names = {name: f"opaque_{salt}_sym{index:03d}" for index, name in enumerate(unique_names)}
    patches: list[tuple[int, int, bytes, str, str]] = []
    records: list[dict[str, object]] = []
    families: set[str] = set()

    for index, node in enumerate(identifiers):
        original = original_names[index]
        if index % 2 == 0:
            replacement = masked_names[original]
            family = "semantic_identifier_masking"
        else:
            replacement = f"opaque_{salt}_{index:03d}"
            family = "identity_fragmentation"
        patches.append((node.start_byte, node.end_byte, replacement.encode(), family, node.type))
        families.add(family)

    # Select maximal semantic-cue nodes so patches never overlap their children.
    semantic_nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "comment" or node.type in STRING_TYPES or node.type in NUMBER_TYPES:
            semantic_nodes.append(node)
            continue
        stack.extend(reversed(node.children))
    for serial, node in enumerate(semantic_nodes):
        original = data[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if node.type == "comment":
            line_count = original.count("\n") + 1
            if original.lstrip().startswith("/*"):
                replacement = "/*" + ("\n" * (line_count - 1)) + f" opaque semantic cue {serial:03d} */"
            else:
                marker = "#" if language == "python" else "//"
                original_lines = original.split("\n")
                replacement_lines = []
                for line in original_lines:
                    continuation = " \\" if line.rstrip().endswith("\\") else ""
                    replacement_lines.append(f"{marker} opaque semantic cue {serial:03d}{continuation}")
                replacement = "\n".join(replacement_lines)
            family = "semantic_cue_erasure"
        elif node.type in STRING_TYPES:
            replacement = replacement_for_string(node.type, original, serial)
            family = "literal_corruption"
        else:
            replacement = str(31337 + serial * 97)
            family = "literal_corruption"
        patches.append((node.start_byte, node.end_byte, replacement.encode(), family, node.type))
        families.add(family)

    covered = [(start, end) for start, end, *_ in patches]
    for node in nodes(root):
        if node.children:
            continue
        if node.parent is None or node.parent.type not in OPERATOR_PARENT_TYPES:
            continue
        token = data[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        if token not in OPERATOR_MAP:
            continue
        if any(start <= node.start_byte and node.end_byte <= end for start, end in covered):
            continue
        replacement = OPERATOR_MAP[token]
        patches.append((node.start_byte, node.end_byte, replacement.encode(), "operator_inversion", node.type))
        families.add("operator_inversion")

    # Reject overlapping patches, then retain the largest parser-safe family combination.
    patches.sort(key=lambda item: (item[0], item[1]))
    for left, right in zip(patches, patches[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping mutation spans in {base_id}: {left[:2]} and {right[:2]}")
    baseline_errors = int(parse_stats(language, source)["error_nodes"])
    ordered_families = sorted(families)
    selected_patches = None
    selected_families = None
    identifier_families = {"semantic_identifier_masking", "identity_fragmentation"}
    candidates = []
    for size in range(len(ordered_families), 1, -1):
        for combination in itertools.combinations(ordered_families, size):
            chosen = set(combination)
            if not chosen.intersection(identifier_families):
                continue
            candidate_patches = [patch for patch in patches if patch[3] in chosen]
            candidates.append((len(chosen), len(candidate_patches), chosen, candidate_patches))
    for _family_count, _patch_count, chosen, candidate_patches in sorted(candidates, reverse=True, key=lambda item: (item[0], item[1])):
        mutated = data
        for start, end, replacement, _family, _node_type in reversed(candidate_patches):
            mutated = mutated[:start] + replacement + mutated[end:]
        candidate = mutated.decode("utf-8")
        if int(parse_stats(language, candidate)["error_nodes"]) <= baseline_errors:
            selected_patches, selected_families = candidate_patches, chosen
            break
    if selected_patches is None or selected_families is None:
        raise ValueError(f"no parser-safe multi-family mutation for {base_id}")
    mutated = data
    for start, end, replacement, family, node_type in reversed(selected_patches):
        original = data[start:end]
        mutated = mutated[:start] + replacement + mutated[end:]
        records.append({
            "family": family, "node_type": node_type, "start_byte_original": start,
            "end_byte_original": end, "original": original.decode("utf-8", errors="replace"),
            "replacement": replacement.decode("utf-8", errors="replace"),
        })
    result = mutated.decode("utf-8")
    result, lexical_comment_changes = scrub_lexical_comments(language, result)
    if lexical_comment_changes:
        records.append({
            "family": "semantic_cue_erasure", "node_type": "pygments_comment_fallback",
            "start_byte_original": -1, "end_byte_original": -1,
            "original": f"{lexical_comment_changes} residual comment token(s)",
            "replacement": "opaque semantic cue",
        })
        selected_families.add("semantic_cue_erasure")
    if int(parse_stats(language, result)["error_nodes"]) > baseline_errors:
        raise ValueError(f"lexical comment scrub increased parser errors for {base_id}")
    return result, sorted(records, key=lambda item: int(item["start_byte_original"])), sorted(selected_families)


def image_stats(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        return {
            "image_width": rgb.width, "image_height": rgb.height,
            "pixel_std": float(np.mean(stat.stddev)), "nonblank": max(stat.stddev) > 0.5,
            "image_sha256": sha256_bytes(path.read_bytes()),
        }


def render_variants(base: pd.DataFrame, variant_source: dict[tuple[str, str], str]) -> pd.DataFrame:
    font = ROOT / "ase-2026-v1/rendering/fonts/DejaVuSansMono.ttf"
    records = []
    for row in base.itertuples(index=False):
        for variant in ("golden", "ugly_gold", "beautiful_trash", "ugly_trash"):
            semantic = "gold" if variant in {"golden", "ugly_gold"} else "trash"
            visual = "clean" if variant in {"golden", "beautiful_trash"} else "ugly"
            code = variant_source[(row.base_id, semantic)]
            path = OUT / "rq3/rendered" / variant / f"{row.base_id}.png"
            if visual == "clean":
                layout = render_code(code, row.language, path, font, "monokai", 20, 80,
                                     line_numbers=True, tab_width=4, image_pad=10, line_gap=6)
            else:
                layout = render_code(code, row.language, path, font, "bw", 18, 60,
                                     line_numbers=False, tab_width=4, image_pad=8, line_gap=2,
                                     remove_indent=True, remove_blank_lines=True)
            records.append({
                "base_id": row.base_id, "rq0_id": row.rq0_id, "language": row.language,
                "score_stratum": row.score_stratum, "variant": variant,
                "semantic_integrity": semantic, "visual_quality": visual,
                "source_sha256": sha256_text(code),
                "image_path": str(path.relative_to(ROOT)), **layout, **image_stats(path),
            })
    return pd.DataFrame(records)


def contrast_manifest(base: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("golden", "ugly_gold", "visual_only_gold", "golden"),
        ("golden", "beautiful_trash", "semantic_and_visual_aligned", "golden"),
        ("golden", "ugly_trash", "semantic_and_visual_aligned", "golden"),
        ("ugly_gold", "beautiful_trash", "semantic_visual_conflict_primary", "ugly_gold"),
        ("ugly_gold", "ugly_trash", "semantic_only_under_ugly", "ugly_gold"),
        ("beautiful_trash", "ugly_trash", "visual_only_trash", "beautiful_trash"),
    ]
    rows = []
    for row in base.itertuples(index=False):
        for index, (left, right, contrast_type, target) in enumerate(specifications):
            rows.append({
                "contrast_id": f"{row.base_id}__c{index + 1}", "base_id": row.base_id,
                "rq0_id": row.rq0_id, "language": row.language, "score_stratum": row.score_stratum,
                "variant_i": left, "variant_j": right, "contrast_type": contrast_type,
                "preference_target_variant": target,
            })
    return pd.DataFrame(rows)


def review_package(base: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for language, frame in base.groupby("language", sort=True):
        rng = random.Random(SEED + {"java": 11, "python": 12, "cuda": 13}[language])
        for stratum, count in (("low", 7), ("mid", 7), ("high", 6)):
            pool = frame[frame.score_stratum.eq(stratum)].index.tolist()
            selected.extend(rng.sample(pool, count))
    review = base.loc[selected, ["base_id", "rq0_id", "language", "score_stratum"]].copy()
    review = review.sort_values(["language", "score_stratum", "base_id"]).reset_index(drop=True)
    review.insert(0, "review_id", [f"gc1_{index:03d}" for index in range(len(review))])
    for column in ["gold_coherent", "trash_incoherent", "clean_legible", "ugly_degraded_no_corruption",
                   "same_tokens_clean_ugly", "approve", "review_notes"]:
        review[column] = ""
    review[["review_id", "base_id", "rq0_id", "language", "score_stratum"]].to_csv(
        OUT / "rq3/review/GC1_SAMPLE.csv", index=False
    )
    review.to_csv(OUT / "rq3/review/GC1_REVIEW_FORM.csv", index=False)
    lookup = metadata.set_index(["base_id", "variant"]).image_path.to_dict()
    review_dir = OUT / "rq3/review"
    cards = []
    for row in review.itertuples(index=False):
        images = "".join(
            f'<figure><img src="{html.escape(os.path.relpath(ROOT / lookup[(row.base_id, variant)], review_dir))}">'
            f'<figcaption>{variant}</figcaption></figure>'
            for variant in ("golden", "ugly_gold", "beautiful_trash", "ugly_trash")
        )
        cards.append(f'<section><h2>{row.review_id}: {row.language} / {row.score_stratum} / {row.base_id}</h2><div>{images}</div></section>')
    page = """<!doctype html><meta charset="utf-8"><title>RQ3 GC1 review</title>
<style>body{font:14px sans-serif;margin:20px;background:#eee;color:#111}section{background:white;margin:0 0 24px;padding:12px;border:1px solid #aaa}h2{font-size:16px}section>div{display:grid;grid-template-columns:1fr 1fr;gap:10px}figure{margin:0;min-width:0}img{max-width:100%;max-height:650px;object-fit:contain;border:1px solid #777;background:#fff}figcaption{font-weight:bold;margin-top:4px}@media(max-width:900px){section>div{grid-template-columns:1fr}}</style>
<h1>RQ3 GC1 author review: 20 items per language</h1>
<p>Record decisions in GC1_REVIEW_FORM.csv. Approve only when gold is coherent, trash intent is not recoverable, clean is legible, ugly is degraded without overlap/clipping, and each clean/ugly pair has identical code content.</p>""" + "".join(cards)
    (OUT / "rq3/review/GC1_REVIEW.html").write_text(page, encoding="utf-8")
    return review


def main() -> int:
    data_dir = OUT / "rq3/data"
    audit_dir = OUT / "rq3/audit"
    review_dir = OUT / "rq3/review"
    for path in (data_dir, audit_dir, review_dir):
        path.mkdir(parents=True, exist_ok=True)
    # Any regenerated source or image invalidates a prior human approval.
    (review_dir / "GC1_APPROVAL.json").unlink(missing_ok=True)
    snippets = pd.read_csv(OUT / "data/render_snippets.csv")
    base, eligibility = select_bases(snippets)
    eligibility.to_csv(audit_dir / "base_eligibility_all_snippets.csv", index=False)
    base.to_csv(data_dir / "rq3_grounded_bases_seed42.csv", index=False)

    source_rows, mutation_rows, variant_source = [], [], {}
    failures = []
    identifier_residuals = []
    for row in base.itertuples(index=False):
        gold = normalize_code(row.raw_code)
        trash, mutations, families = mutate(row.language, gold, row.base_id)
        before, after = parse_stats(row.language, gold), parse_stats(row.language, trash)
        gold_bytes, trash_bytes = gold.encode(), trash.encode()
        gold_identifiers = {
            gold_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            for node in nodes(PARSERS[row.language].parse(gold_bytes).root_node)
            if node.type in ID_TYPES and not node.children
        }
        trash_identifiers = {
            trash_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            for node in nodes(PARSERS[row.language].parse(trash_bytes).root_node)
            if node.type in ID_TYPES and not node.children
        }
        residual = sorted(value for value in gold_identifiers & trash_identifiers if value and not value.startswith("opaque_"))
        if residual:
            identifier_residuals.append({"base_id": row.base_id, "identifiers": residual})
        passed = after["error_nodes"] <= before["error_nodes"] and len(families) >= 2 and gold != trash
        if not passed:
            failures.append({"base_id": row.base_id, "before": before, "after": after, "families": families})
        variant_source[(row.base_id, "gold")] = gold
        variant_source[(row.base_id, "trash")] = trash
        source_rows.extend([
            {"base_id": row.base_id, "rq0_id": row.rq0_id, "language": row.language, "semantic_integrity": "gold",
             "code": gold, "source_sha256": sha256_text(gold), **{f"parse_{k}": v for k, v in before.items()}},
            {"base_id": row.base_id, "rq0_id": row.rq0_id, "language": row.language, "semantic_integrity": "trash",
             "code": trash, "source_sha256": sha256_text(trash), **{f"parse_{k}": v for k, v in after.items()}},
        ])
        for mutation in mutations:
            mutation_rows.append({"base_id": row.base_id, "rq0_id": row.rq0_id, "language": row.language, **mutation})
    if failures or identifier_residuals:
        payload = {"parser_failures": failures, "identifier_residuals": identifier_residuals}
        (audit_dir / "mutation_failures.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(f"mutation gate failed: parser={len(failures)}, identifiers={len(identifier_residuals)}")
    sources = pd.DataFrame(source_rows)
    mutations = pd.DataFrame(mutation_rows)
    sources.to_csv(data_dir / "rq3_variant_sources.csv", index=False)
    mutations.to_csv(data_dir / "rq3_mutation_log.csv", index=False)

    metadata = render_variants(base, variant_source)
    metadata.to_csv(data_dir / "rq3_render_metadata.csv", index=False)
    contrasts = contrast_manifest(base)
    contrasts.to_csv(data_dir / "rq3_contrast_manifest.csv", index=False)
    review = review_package(base, metadata)

    family_counts = mutations.groupby(["language", "family"]).size().reset_index(name="mutations")
    family_counts.to_csv(audit_dir / "mutation_family_counts.csv", index=False)
    wide_hash = metadata.pivot(index="base_id", columns="variant", values="source_sha256")
    visual_pair_hash_ok = (wide_hash.golden.eq(wide_hash.ugly_gold) & wide_hash.beautiful_trash.eq(wide_hash.ugly_trash))
    image_pair_distinct = []
    by_image = metadata.set_index(["base_id", "variant"])
    for base_id in base.base_id:
        image_pair_distinct.append(
            by_image.loc[(base_id, "golden"), "image_sha256"] != by_image.loc[(base_id, "ugly_gold"), "image_sha256"]
            and by_image.loc[(base_id, "beautiful_trash"), "image_sha256"] != by_image.loc[(base_id, "ugly_trash"), "image_sha256"]
        )
    family_per_base = mutations.groupby("base_id").family.nunique()
    manifest = {
        "status": "prepared_pending_author_gc1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bases": len(base), "bases_by_language": base.groupby("language").size().to_dict(),
        "score_strata_by_language": {f"{a}/{b}": int(v) for (a, b), v in base.groupby(["language", "score_stratum"]).size().items()},
        "variant_sources": len(sources), "rendered_images": len(metadata), "contrasts": len(contrasts),
        "contrasts_by_language": contrasts.groupby("language").size().to_dict(),
        "minimum_mutation_families_per_base": int(family_per_base.min()),
        "all_parse_error_counts_nonincreasing": True,
        "all_original_identifier_cues_removed": True,
        "all_visual_source_hash_pairs_match": bool(visual_pair_hash_ok.all()),
        "all_clean_ugly_image_pairs_distinct": bool(all(image_pair_distinct)),
        "all_images_nonblank": bool(metadata.nonblank.all()),
        "gc1_review_items": len(review), "gc1_by_language": review.groupby("language").size().to_dict(),
        "inference_authorized": False,
        "inference_blocker": "GC1_REVIEW_FORM.csv requires author approval for all 60 sampled bases",
        "parser_versions": {"tree_sitter": "0.25.2", "java": "0.23.5", "python": "0.25.0", "cpp_for_cuda": "0.23.4"},
    }
    gate = (
        len(base) == 300 and len(sources) == 600 and len(metadata) == 1200 and len(contrasts) == 1800
        and manifest["minimum_mutation_families_per_base"] >= 2
        and manifest["all_visual_source_hash_pairs_match"] and manifest["all_clean_ugly_image_pairs_distinct"]
        and manifest["all_images_nonblank"] and len(review) == 60
    )
    manifest["automatic_preparation_gate_pass"] = bool(gate)
    (OUT / "rq3/RQ3_PREPARATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())

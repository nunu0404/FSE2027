#!/usr/bin/env python3
"""Build and audit the frozen 9,000-pair RQ1 model-battery assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/rq1_model_battery_3lang_20260723"
JAVA_PAIRS = ROOT / "experiments/rq0_viability/data/pairs/full_pair_set_rq0.csv"
EXT_PAIRS = ROOT / "results/python_cuda_vlm_main_20260715/data/pairs_seed42_clean.csv"
GROUNDED_PAIRS = ROOT / "results/grounded_protocol_3lang_20260721/data/pairs_seed42_grounded.csv"
RENDER_META = ROOT / "results/grounded_protocol_3lang_20260721/rendered/metadata/grid_render_metadata.csv"
RENDER_AUDIT = ROOT / "results/grounded_protocol_3lang_20260721/audit/GROUNDED_RENDER_AUDIT.json"
JAVA_REG = ROOT / "experiments/rq0_viability/outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv"
JAVA_CANON = ROOT / "experiments/rq0_viability/outputs/canonical_baselines/canonical_pair_level.csv"
EXT_ML = ROOT / "results/dataset_extension_20260713/data/pair_predictions.csv"
PROMPT = ROOT / "results/grounded_protocol_3lang_20260721/config/prompt_B_template.txt"
BASELINE = "monokai_dark__fs20__wrap80__lnon"
LANGUAGES = ["java", "python", "cuda"]
DIFFICULTIES = ["easy", "medium", "hard"]
MODELS = [
    "Logistic Regression",
    "Multilayer Perceptron",
    "Naive Bayes",
    "Scalabrino-LR (replicated, not official tool)",
    "Voting ensemble (LR+NB+RF)",
    "gradient_boosting_regressor",
    "linear_regression",
    "random_forest_regressor",
    "svr",
]
GA0_EXTRA_CELLS = {
    ("java", "easy"),
    ("java", "medium"),
    ("python", "hard"),
    ("python", "easy"),
    ("cuda", "medium"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalize_pairs() -> pd.DataFrame:
    java = pd.read_csv(JAVA_PAIRS)
    java["language"] = "java"
    ext = pd.read_csv(EXT_PAIRS)
    require(len(java) == 3000, "Java pair count is not 3,000")
    require(ext.groupby("language").size().to_dict() == {"cuda": 3000, "python": 3000},
            "Python/CUDA pair allocation drift")

    columns = [
        "pair_id", "snippet_i", "snippet_j", "dataset_name_i", "dataset_name_j",
        "language", "difficulty", "human_score_i_z", "human_score_j_z",
        "abs_z_diff", "human_preference", "preference_score_basis",
    ]
    pairs = pd.concat([java[columns], ext[columns]], ignore_index=True)
    pairs.insert(0, "protocol_pair_id", [f"rq1_{i:04d}" for i in range(len(pairs))])
    pairs["seed"] = 42
    pairs["difficulty_role"] = "legacy_sampling_stratum"
    pairs["continuous_difficulty"] = pairs["abs_z_diff"]

    require(len(pairs) == 9000, "combined pair count is not 9,000")
    require(pairs["pair_id"].is_unique, "source pair IDs are not globally unique")
    require(pairs["protocol_pair_id"].is_unique, "protocol pair IDs are not unique")
    require(not pairs[["snippet_i", "snippet_j", "human_preference"]].isna().any().any(),
            "required pair field is missing")
    require((pairs["snippet_i"] != pairs["snippet_j"]).all(), "self-pair found")

    expected = {(language, difficulty): 1000 for language in LANGUAGES for difficulty in DIFFICULTIES}
    allocation = pairs.groupby(["language", "difficulty"]).size().to_dict()
    require(allocation == expected, f"language/difficulty allocation drift: {allocation}")

    expected_gold = np.where(
        pairs["human_score_i_z"] > pairs["human_score_j_z"],
        pairs["snippet_i"],
        pairs["snippet_j"],
    )
    require((pairs["human_score_i_z"] != pairs["human_score_j_z"]).all(), "human-score tie found")
    require(np.array_equal(expected_gold, pairs["human_preference"].to_numpy()),
            "human preference disagrees with z-score direction")
    recomputed = (pairs["human_score_i_z"] - pairs["human_score_j_z"]).abs()
    require(np.allclose(recomputed, pairs["abs_z_diff"], rtol=0, atol=1e-12),
            "abs_z_diff does not match score difference")

    unordered = pairs.apply(
        lambda row: (row["language"], *sorted((row["snippet_i"], row["snippet_j"]))),
        axis=1,
    )
    require(not unordered.duplicated().any(), "duplicate unordered pair found")
    return pairs


def attach_and_audit_images(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    existing_audit = json.loads(RENDER_AUDIT.read_text(encoding="utf-8"))
    require(bool(existing_audit.get("gate_pass")), "upstream grounded render audit did not pass")

    metadata = pd.read_csv(RENDER_META)
    baseline = metadata[metadata["condition"].eq(BASELINE)].copy()
    referenced = set(pairs["snippet_i"]) | set(pairs["snippet_j"])
    require(len(referenced) == 552, f"expected 552 referenced snippets, found {len(referenced)}")
    require(baseline["rq0_id"].is_unique, "baseline render metadata has duplicate snippet IDs")
    lookup = baseline.set_index("rq0_id")
    missing = sorted(referenced - set(lookup.index))
    require(not missing, f"missing baseline render metadata: {missing[:5]}")

    audit_rows: list[dict[str, object]] = []
    for snippet_id in sorted(referenced):
        row = lookup.loc[snippet_id]
        relative = str(row["image_path"])
        path = ROOT / relative
        require(path.is_file(), f"missing rendered PNG: {path}")
        actual_hash = sha256(path)
        require(actual_hash == row["image_sha256"], f"PNG hash mismatch: {snippet_id}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            extrema = rgb.getextrema()
            std_mean = float(np.mean(ImageStat.Stat(rgb).stddev))
            require(rgb.width == int(row["image_width"]) and rgb.height == int(row["image_height"]),
                    f"PNG dimensions disagree with metadata: {snippet_id}")
            require(std_mean > 0 and any(low != high for low, high in extrema),
                    f"blank/constant PNG: {snippet_id}")
            audit_rows.append({
                "snippet_id": snippet_id,
                "language": row["language"],
                "image_path": relative,
                "image_sha256": actual_hash,
                "width": rgb.width,
                "height": rgb.height,
                "pixel_std_mean": std_mean,
                "nonconstant": True,
            })

    image_audit = pd.DataFrame(audit_rows)
    image_lookup = image_audit.set_index("snippet_id")
    pairs = pairs.copy()
    pairs["image_i_path"] = pairs["snippet_i"].map(image_lookup["image_path"])
    pairs["image_j_path"] = pairs["snippet_j"].map(image_lookup["image_path"])
    pairs["image_i_sha256"] = pairs["snippet_i"].map(image_lookup["image_sha256"])
    pairs["image_j_sha256"] = pairs["snippet_j"].map(image_lookup["image_sha256"])
    require(not pairs[["image_i_path", "image_j_path"]].isna().any().any(),
            "pair-to-image join produced missing values")

    image_audit.to_csv(OUT / "audit/image_integrity_552.csv", index=False)
    return pairs, {
        "upstream_render_audit": str(RENDER_AUDIT.relative_to(ROOT)),
        "upstream_render_audit_sha256": sha256(RENDER_AUDIT),
        "upstream_gate_pass": True,
        "baseline_condition": BASELINE,
        "referenced_snippets": len(referenced),
        "png_files_checked": len(image_audit),
        "png_hash_matches": int(len(image_audit)),
        "blank_or_constant_pngs": 0,
        "languages": image_audit.groupby("language").size().to_dict(),
    }


def normalize_ml_predictions(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    for source in [JAVA_REG, JAVA_CANON]:
        frame = pd.read_csv(source)
        frame = frame[[
            "pair_id", "model", "model_preference", "is_correct",
            "predicted_score_i", "predicted_score_j",
        ]].copy()
        frame.insert(0, "language", "java")
        frame["prediction_tie"] = frame["predicted_score_i"].eq(frame["predicted_score_j"])
        frame["source_file"] = str(source.relative_to(ROOT))
        frames.append(frame)

    ext = pd.read_csv(EXT_ML)
    ext = ext[ext["language"].isin(["python", "cuda"])].copy()
    ext_pairs = pairs[pairs["language"].isin(["python", "cuda"])][
        ["pair_id", "snippet_i", "snippet_j", "human_preference"]
    ]
    ext = ext.merge(ext_pairs, on="pair_id", how="left", validate="many_to_one")
    require(not ext[["snippet_i", "snippet_j", "human_preference"]].isna().any().any(),
            "extended ML predictions do not join to RQ1 pairs")
    other = np.where(
        ext["human_preference"].eq(ext["snippet_i"]),
        ext["snippet_j"],
        ext["snippet_i"],
    )
    ext["model_preference"] = np.where(
        ext["prediction_tie"],
        None,
        np.where(ext["correct"], ext["human_preference"], other),
    )
    ext["is_correct"] = ext["correct"].astype(bool) & ~ext["prediction_tie"].astype(bool)
    ext["predicted_score_i"] = np.where(
        ext["human_preference"].eq(ext["snippet_i"]),
        ext["winner_predicted_z"],
        ext["loser_predicted_z"],
    )
    ext["predicted_score_j"] = np.where(
        ext["human_preference"].eq(ext["snippet_j"]),
        ext["winner_predicted_z"],
        ext["loser_predicted_z"],
    )
    ext["source_file"] = str(EXT_ML.relative_to(ROOT))
    frames.append(ext[[
        "language", "pair_id", "model", "model_preference", "is_correct",
        "predicted_score_i", "predicted_score_j", "prediction_tie", "source_file",
    ]])

    predictions = pd.concat(frames, ignore_index=True)
    require(len(predictions) == 81000, f"expected 81,000 ML rows, found {len(predictions)}")
    require(not predictions.duplicated(["language", "pair_id", "model"]).any(),
            "duplicate language/pair/model ML prediction")
    require(set(predictions["model"]) == set(MODELS), "ML model set drift")

    pair_sets = {
        language: set(pairs.loc[pairs["language"].eq(language), "pair_id"])
        for language in LANGUAGES
    }
    coverage_rows: list[dict[str, object]] = []
    for language in LANGUAGES:
        for model in MODELS:
            observed = set(predictions.loc[
                predictions["language"].eq(language) & predictions["model"].eq(model),
                "pair_id",
            ])
            missing = pair_sets[language] - observed
            extra = observed - pair_sets[language]
            coverage_rows.append({
                "language": language,
                "model": model,
                "expected_pairs": 3000,
                "observed_pairs": len(observed),
                "missing_pairs": len(missing),
                "extra_pairs": len(extra),
                "exact_match": not missing and not extra,
            })
    coverage = pd.DataFrame(coverage_rows)
    require(coverage["exact_match"].all(), "ML pair-ID coverage is not exact")
    coverage.to_csv(OUT / "audit/ml_pair_id_coverage.csv", index=False)
    return predictions, {
        "models": MODELS,
        "prediction_rows": len(predictions),
        "expected_prediction_rows": 81000,
        "language_model_cells": len(coverage),
        "exact_pair_id_cells": int(coverage["exact_match"].sum()),
        "all_cells_exact": True,
        "ties_by_language": predictions.groupby("language")["prediction_tie"].sum().astype(int).to_dict(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [JAVA_REG, JAVA_CANON, EXT_ML]
        },
    }


def audit_grounded_subset(pairs: pd.DataFrame) -> dict[str, object]:
    grounded = pd.read_csv(GROUNDED_PAIRS)
    full_ids = set(pairs["pair_id"])
    subset_ids = set(grounded["pair_id"])
    require(len(grounded) == 3000 and subset_ids <= full_ids,
            "grounded 3,000-pair set is not an exact subset of the RQ1 full set")
    return {
        "grounded_rows": len(grounded),
        "grounded_unique_source_pair_ids": len(subset_ids),
        "all_grounded_pairs_in_full_9000": True,
        "grounded_source_sha256": sha256(GROUNDED_PAIRS),
    }


def freeze_ga0_sample(pairs: pd.DataFrame) -> tuple[Path, dict[str, object]]:
    rng = np.random.default_rng(42)
    selected: list[pd.DataFrame] = []
    for language in LANGUAGES:
        for difficulty in DIFFICULTIES:
            cell = pairs[
                pairs["language"].eq(language) & pairs["difficulty"].eq(difficulty)
            ].sort_values("pair_id")
            count = 6 if (language, difficulty) in GA0_EXTRA_CELLS else 5
            positions = np.sort(rng.choice(len(cell), size=count, replace=False))
            selected.append(cell.iloc[positions])
    ga0 = pd.concat(selected, ignore_index=True).sort_values(
        ["language", "difficulty", "pair_id"]
    )
    require(len(ga0) == 50 and ga0["pair_id"].is_unique, "GA0 sample allocation failed")
    ga0_path = OUT / "data/ga0_pairs_50_seed42.csv"
    ga0.to_csv(ga0_path, index=False)
    return ga0_path, {
        "pairs": len(ga0),
        "calls_per_model": 2 * len(ga0),
        "seed": 42,
        "allocation_by_language": ga0.groupby("language").size().to_dict(),
        "allocation_by_difficulty": ga0.groupby("difficulty").size().to_dict(),
        "allocation_by_cell": {
            f"{language}/{difficulty}": int(count)
            for (language, difficulty), count in ga0.groupby(["language", "difficulty"]).size().items()
        },
    }


def main() -> int:
    for directory in [OUT / "data", OUT / "audit"]:
        directory.mkdir(parents=True, exist_ok=True)

    pairs = normalize_pairs()
    pairs, image_summary = attach_and_audit_images(pairs)
    predictions, ml_summary = normalize_ml_predictions(pairs)
    subset_summary = audit_grounded_subset(pairs)
    ga0_path, ga0_summary = freeze_ga0_sample(pairs)

    pair_path = OUT / "data/rq1_pairs_9000.csv"
    ml_path = OUT / "data/ml_predictions_9models_9000.csv"
    pairs.to_csv(pair_path, index=False)
    predictions.to_csv(ml_path, index=False)

    evidence_files = sorted((OUT / "evidence/sources").glob("*"))
    manifest = {
        "status": "PASS",
        "scope": "preregistration assets only; no new VLM outcomes generated",
        "pair_rows": len(pairs),
        "pairs_by_language": pairs.groupby("language").size().to_dict(),
        "pairs_by_language_difficulty": {
            f"{language}/{difficulty}": int(count)
            for (language, difficulty), count in pairs.groupby(["language", "difficulty"]).size().items()
        },
        "unique_snippets": len(set(pairs["snippet_i"]) | set(pairs["snippet_j"])),
        "gold_direction_matches_z": True,
        "duplicate_unordered_pairs": 0,
        "continuous_primary_difficulty": "abs_z_diff",
        "legacy_sampling_bins": {
            "hard": "0.2 <= abs_z_diff < 0.5",
            "medium": "0.5 <= abs_z_diff < 1.0",
            "easy": "abs_z_diff >= 1.0",
        },
        "image_audit": image_summary,
        "ml_prediction_audit": ml_summary,
        "grounded_subset_audit": subset_summary,
        "ga0_sample": ga0_summary,
        "artifacts": {
            str(pair_path.relative_to(ROOT)): sha256(pair_path),
            str(ml_path.relative_to(ROOT)): sha256(ml_path),
            str(ga0_path.relative_to(ROOT)): sha256(ga0_path),
            str(PROMPT.relative_to(ROOT)): sha256(PROMPT),
        },
        "archived_external_sources": {
            str(path.relative_to(ROOT)): sha256(path) for path in evidence_files if path.is_file()
        },
        "source_pair_sha256": {
            str(JAVA_PAIRS.relative_to(ROOT)): sha256(JAVA_PAIRS),
            str(EXT_PAIRS.relative_to(ROOT)): sha256(EXT_PAIRS),
        },
    }
    manifest_path = OUT / "audit/PREREG_ASSET_AUDIT.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

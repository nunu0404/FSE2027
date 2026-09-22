#!/usr/bin/env python3
"""Build and evaluate the zero-cost B3 stage-1 dataset extension.

This script reads immutable official archives extracted under data/raw_ratings,
builds Dorn-360 and Sergeyuk-120, audits baseline renders, extracts the same
27-feature family as RQ0, and runs seed-42 five-fold classical regressors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_features import extract_one  # noqa: E402
from render_code_images import wrap_code  # noqa: E402
from run_canonical_baselines import model_defs as canonical_model_defs  # noqa: E402
from train_classical_baselines import regression_models  # noqa: E402


SEED = 42
EXPECTED_DORN = {"java": 121, "python": 119, "cuda": 120}
VALID_SERGEYUK_LABELS = {"Readable", "Unreadable"}


def sha256_text(value: str, remove_space: bool = False) -> str:
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    if remove_space:
        value = "".join(value.split())
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    std = values.std(ddof=0)
    if std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def load_dorn(raw_root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    root = raw_root / "extracted/DatasetDorn/dataset"
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for language, expected in EXPECTED_DORN.items():
        snippet_dir = root / "snippets" / language
        paths = sorted(snippet_dir.glob("*.jsnp"), key=lambda p: int(p.stem))
        ratings = pd.read_csv(root / "scores" / f"{language}.csv", header=None)
        participant_ids = ratings.iloc[:, 0]
        matrix = ratings.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        if len(paths) != expected or matrix.shape[1] != expected:
            raise RuntimeError(
                f"Dorn {language} size mismatch: snippets={len(paths)} rating_cols={matrix.shape[1]} expected={expected}"
            )
        if participant_ids.nunique() != len(participant_ids):
            raise RuntimeError(f"Dorn {language} participant IDs are not unique")
        for column, path in enumerate(paths):
            values = matrix.iloc[:, column].dropna().astype(float)
            if values.empty:
                raise RuntimeError(f"Dorn {language}/{path.stem} has no ratings")
            rows.append(
                {
                    "snippet_uid": f"dorn_{language}_{int(path.stem):03d}",
                    "dataset_name": "Dorn",
                    "language": language,
                    "original_snippet_id": path.stem,
                    "raw_code": path.read_text(encoding="utf-8", errors="replace"),
                    "human_mean_score": float(values.mean()),
                    "rating_count": int(values.count()),
                    "readable_votes": np.nan,
                    "unreadable_votes": np.nan,
                    "invalid_votes_excluded": 0,
                    "official_ambiguous": False,
                }
            )
        checks.append(
            {
                "dataset_name": "Dorn",
                "language": language,
                "snippet_count": len(paths),
                "participant_rows": len(ratings),
                "ratings_total": int(matrix.count().sum()),
                "ratings_per_snippet_min": int(matrix.count().min()),
                "ratings_per_snippet_median": float(matrix.count().median()),
                "ratings_per_snippet_max": int(matrix.count().max()),
                "score_scale": "1..5 Likert",
                "score_column_mapping": "numeric snippet IDs sorted ascending; positional score columns",
            }
        )
    return pd.DataFrame(rows), checks


def load_sergeyuk(raw_root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    root = raw_root / "extracted/sergeyuk_readability/data"
    aggregated = pd.read_csv(root / "aggregated.csv")
    snippets = pd.read_csv(root / "snippets.csv")
    valid = aggregated[aggregated["readability"].isin(VALID_SERGEYUK_LABELS)].copy()
    valid["snip_id"] = valid["snippet"].str.removesuffix(".png")
    vote_counts = valid.groupby("snip_id")["readability"].agg(
        rating_count="size",
        readable_votes=lambda x: int((x == "Readable").sum()),
        unreadable_votes=lambda x: int((x == "Unreadable").sum()),
    )
    all_counts = aggregated.assign(snip_id=aggregated["snippet"].str.removesuffix(".png")).groupby("snip_id").size()
    invalid_counts = all_counts.subtract(vote_counts["rating_count"], fill_value=0).astype(int)
    selected = snippets[snippets["snip_id"].astype(str).isin(vote_counts.index)].copy()
    if len(vote_counts) != 120 or len(selected) != 120:
        raise RuntimeError(f"Sergeyuk size mismatch: vote snippets={len(vote_counts)} code snippets={len(selected)}")
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        snip_id = str(row.snip_id)
        counts = vote_counts.loc[snip_id]
        n = int(counts.rating_count)
        readable = int(counts.readable_votes)
        unreadable = int(counts.unreadable_votes)
        rows.append(
            {
                "snippet_uid": f"sergeyuk_java_{snip_id}",
                "dataset_name": "Sergeyuk2024",
                "language": "java",
                "original_snippet_id": snip_id,
                "raw_code": str(row.snippet),
                "human_mean_score": readable / n,
                "rating_count": n,
                "readable_votes": readable,
                "unreadable_votes": unreadable,
                "invalid_votes_excluded": int(invalid_counts.loc[snip_id]),
                "official_ambiguous": abs(readable - unreadable) / n < 0.1,
            }
        )
    checks = [
        {
            "dataset_name": "Sergeyuk2024",
            "language": "java",
            "snippet_count": len(rows),
            "participant_rows": int(aggregated["response_id"].nunique()),
            "ratings_total": int(len(valid)),
            "ratings_per_snippet_min": int(vote_counts.rating_count.min()),
            "ratings_per_snippet_median": float(vote_counts.rating_count.median()),
            "ratings_per_snippet_max": int(vote_counts.rating_count.max()),
            "score_scale": "readable vote proportion [0,1]",
            "invalid_category_rows_excluded": int(len(aggregated) - len(valid)),
            "official_ambiguous_snippets": int(sum(r["official_ambiguous"] for r in rows)),
            "score_column_mapping": "snip_id join after removing .png suffix",
        }
    ]
    return pd.DataFrame(rows), checks


def finalize_dataset(dorn: pd.DataFrame, sergeyuk: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([dorn, sergeyuk], ignore_index=True)
    data["human_score_z_within_dataset"] = data.groupby("dataset_name", group_keys=False)["human_mean_score"].transform(zscore)
    data["human_score_z_within_language"] = data.groupby(
        ["dataset_name", "language"], group_keys=False
    )["human_mean_score"].transform(zscore)
    data["exact_code_hash"] = data["raw_code"].map(sha256_text)
    data["formatting_code_hash"] = data["raw_code"].map(lambda value: sha256_text(value, remove_space=True))
    return data.sort_values(["dataset_name", "language", "original_snippet_id"]).reset_index(drop=True)


def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in data.itertuples(index=False):
        values = extract_one(row.raw_code, row.language)
        values.update(
            {
                "snippet_uid": row.snippet_uid,
                "dataset_name": row.dataset_name,
                "language": row.language,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def render_and_audit(data: pd.DataFrame, repo: Path, output: Path) -> pd.DataFrame:
    config = yaml.safe_load((repo / "experiments/rq0_viability/configs/rq0_main.yaml").read_text(encoding="utf-8"))
    rendering = config["rendering"]["default"]
    font_path = repo / rendering["font_name"]
    image_dir = output / "rendered/default"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in data.itertuples(index=False):
        path = image_dir / f"{row.snippet_uid}.png"
        wrapped = wrap_code(row.raw_code, int(rendering["wrap_column"]))
        success = True
        error = ""
        width = height = edge_anomaly_pixels = 0
        try:
            try:
                lexer = get_lexer_by_name(row.language)
            except ClassNotFound:
                lexer = TextLexer()
            formatter = ImageFormatter(
                font_name=str(font_path),
                font_size=int(rendering["font_size"]),
                line_numbers=bool(rendering["line_numbers"]),
                style=str(rendering["style"]),
                image_format="PNG",
                line_pad=int(rendering["line_pad"]),
                image_pad=int(rendering["image_pad"]),
            )
            path.write_bytes(highlight(wrapped, lexer, formatter))
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
                height, width = rgb.shape[:2]
                # ImageFormatter has separate line-number and code backgrounds.
                # A healthy padded image has uniform left/right edges and an
                # identical background/divider pattern on its top/bottom edges.
                edge_anomaly_pixels = int(
                    np.any(rgb[:, 0] != rgb[0, 0], axis=1).sum()
                    + np.any(rgb[:, -1] != rgb[0, -1], axis=1).sum()
                    + np.any(rgb[0, :] != rgb[-1, :], axis=1).sum()
                )
                if width <= 0 or height <= 0 or edge_anomaly_pixels > 0:
                    raise RuntimeError(
                        f"possible clipping: width={width} height={height} edge_anomalies={edge_anomaly_pixels}"
                    )
        except Exception as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "snippet_uid": row.snippet_uid,
                "dataset_name": row.dataset_name,
                "language": row.language,
                "image_path": str(path.resolve()),
                "width": width,
                "height": height,
                "wrapped_lines": len(wrapped.splitlines()),
                "edge_anomaly_pixels": edge_anomaly_pixels,
                "render_success": success,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def make_folds(group: pd.DataFrame) -> np.ndarray:
    ranked = group["human_mean_score"].rank(method="first")
    strata = pd.qcut(ranked, q=3, labels=["low", "mid", "high"])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.full(len(group), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(
        splitter.split(group, strata, groups=group["formatting_code_hash"])
    ):
        folds[test_idx] = fold
    if (folds < 0).any():
        raise RuntimeError("Incomplete fold assignment")
    return folds


def oof_predictions(data: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = data.merge(features, on=["snippet_uid", "dataset_name", "language"], validate="one_to_one")
    feature_cols = [column for column in merged if column.startswith("feature_")]
    prediction_rows = []
    metric_rows = []
    for (dataset, language), raw_group in merged.groupby(["dataset_name", "language"], sort=True):
        group = raw_group.reset_index(drop=True)
        folds = make_folds(group)
        for model_name in regression_models(SEED):
            predictions = np.full(len(group), np.nan)
            for fold in range(5):
                train = group[folds != fold]
                test = group[folds == fold]
                model = regression_models(SEED)[model_name]
                model.fit(train[feature_cols], train["human_score_z_within_dataset"])
                predictions[folds == fold] = model.predict(test[feature_cols])
            rho, pvalue = spearmanr(group["human_score_z_within_dataset"], predictions)
            metric_rows.append(
                {
                    "dataset_name": dataset,
                    "language": language,
                    "model": model_name,
                    "n_snippets": len(group),
                    "feature_count": len(feature_cols),
                    "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
                    "spearman_r": float(rho),
                    "spearman_p": float(pvalue),
                    "mae_z": float(mean_absolute_error(group["human_score_z_within_dataset"], predictions)),
                    "rmse_z": float(np.sqrt(mean_squared_error(group["human_score_z_within_dataset"], predictions))),
                }
            )
            for index, row in group.iterrows():
                prediction_rows.append(
                    {
                        "snippet_uid": row.snippet_uid,
                        "dataset_name": dataset,
                        "language": language,
                        "model": model_name,
                        "fold": int(folds[index]),
                        "gold_z": float(row.human_score_z_within_dataset),
                        "predicted_z": float(predictions[index]),
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def oof_canonical_predictions(data: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the five canonical binary baselines used in the existing headline table."""
    merged = data.merge(features, on=["snippet_uid", "dataset_name", "language"], validate="one_to_one")
    feature_cols = [column for column in merged if column.startswith("feature_")]
    prediction_rows = []
    metric_rows = []
    for (dataset, language), raw_group in merged.groupby(["dataset_name", "language"], sort=True):
        group = raw_group.reset_index(drop=True)
        folds = make_folds(group)
        threshold = float(group["human_mean_score"].mean())
        target = (group["human_mean_score"] >= threshold).astype(int).to_numpy()
        for model_name in canonical_model_defs(SEED):
            probabilities = np.full(len(group), np.nan)
            labels = np.full(len(group), -1, dtype=int)
            for fold in range(5):
                train_mask = folds != fold
                test_mask = folds == fold
                model = canonical_model_defs(SEED)[model_name]
                model.fit(group.loc[train_mask, feature_cols], target[train_mask])
                probabilities[test_mask] = model.predict_proba(group.loc[test_mask, feature_cols])[:, 1]
                labels[test_mask] = model.predict(group.loc[test_mask, feature_cols]).astype(int)
            metric_rows.append(
                {
                    "dataset_name": dataset,
                    "language": language,
                    "model": model_name,
                    "n_snippets": len(group),
                    "feature_count": len(feature_cols),
                    "threshold": threshold,
                    "target": "human mean >= within-language mean",
                    "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
                    "accuracy": float(accuracy_score(target, labels)),
                    "balanced_accuracy": float(balanced_accuracy_score(target, labels)),
                    "macro_f1": float(f1_score(target, labels, average="macro")),
                    "roc_auc": float(roc_auc_score(target, probabilities)),
                }
            )
            for index, row in group.iterrows():
                prediction_rows.append(
                    {
                        "snippet_uid": row.snippet_uid,
                        "dataset_name": dataset,
                        "language": language,
                        "model": model_name,
                        "fold": int(folds[index]),
                        "binary_target": int(target[index]),
                        "predicted_probability": float(probabilities[index]),
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def sampled_pairs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = []
    availability = []
    for (dataset, language), group in data.groupby(["dataset_name", "language"], sort=True):
        candidates = {"hard": [], "medium": [], "easy": []}
        records = list(group.itertuples(index=False))
        for left, right in combinations(records, 2):
            delta = abs(left.human_score_z_within_dataset - right.human_score_z_within_dataset)
            if delta < 0.2:
                continue
            difficulty = "easy" if delta >= 1.0 else "medium" if delta >= 0.5 else "hard"
            candidates[difficulty].append((left, right, delta))
        for difficulty in ["easy", "medium", "hard"]:
            bucket = candidates[difficulty]
            rng = np.random.default_rng(SEED)
            selected_indices = rng.choice(len(bucket), size=min(1000, len(bucket)), replace=False) if bucket else []
            availability.append(
                {
                    "dataset_name": dataset,
                    "language": language,
                    "difficulty": difficulty,
                    "available_pairs": len(bucket),
                    "sampled_pairs": len(selected_indices),
                }
            )
            for index in selected_indices:
                left, right, delta = bucket[int(index)]
                if left.human_score_z_within_dataset > right.human_score_z_within_dataset:
                    winner, loser = left, right
                else:
                    winner, loser = right, left
                pairs.append(
                    {
                        "pair_id": f"ext_{dataset.lower()}_{language}_{difficulty}_{len(pairs):05d}",
                        "dataset_name": dataset,
                        "language": language,
                        "difficulty": difficulty,
                        "winner_uid": winner.snippet_uid,
                        "loser_uid": loser.snippet_uid,
                        "abs_delta_z": float(delta),
                    }
                )
    return pd.DataFrame(pairs), pd.DataFrame(availability)


def score_pairs(pairs: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = predictions.set_index(["model", "snippet_uid"])["predicted_z"]
    rows = []
    summaries = []
    for model in sorted(predictions.model.unique()):
        for pair in pairs.itertuples(index=False):
            winner_score = float(lookup.loc[(model, pair.winner_uid)])
            loser_score = float(lookup.loc[(model, pair.loser_uid)])
            correct = winner_score > loser_score
            tie = winner_score == loser_score
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "dataset_name": pair.dataset_name,
                    "language": pair.language,
                    "difficulty": pair.difficulty,
                    "model": model,
                    "winner_predicted_z": winner_score,
                    "loser_predicted_z": loser_score,
                    "correct": correct,
                    "prediction_tie": tie,
                }
            )
    scored = pd.DataFrame(rows)
    for keys, group in scored.groupby(["dataset_name", "language", "model", "difficulty"], sort=True):
        dataset, language, model, difficulty = keys
        summaries.append(
            {
                "dataset_name": dataset,
                "language": language,
                "model": model,
                "difficulty": difficulty,
                "n_pairs": len(group),
                "pair_accuracy": float(group.correct.mean()),
                "prediction_ties": int(group.prediction_tie.sum()),
            }
        )
    for keys, group in scored.groupby(["dataset_name", "language", "model"], sort=True):
        dataset, language, model = keys
        summaries.append(
            {
                "dataset_name": dataset,
                "language": language,
                "model": model,
                "difficulty": "all",
                "n_pairs": len(group),
                "pair_accuracy": float(group.correct.mean()),
                "prediction_ties": int(group.prediction_tie.sum()),
            }
        )
    return scored, pd.DataFrame(summaries)


def write_reports(
    output: Path,
    data: pd.DataFrame,
    checks: pd.DataFrame,
    render_audit: pd.DataFrame,
    snippet_metrics: pd.DataFrame,
    canonical_metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
) -> None:
    best = snippet_metrics.sort_values("spearman_r", ascending=False).groupby(
        ["dataset_name", "language"], as_index=False
    ).first()
    pair_all = pair_metrics[pair_metrics.difficulty.eq("all")].copy()
    report = [
        "# Dataset Extension: B3 Stage 1",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope and method",
        "",
        "- Cost/API calls: 0.",
        "- Random seed: 42.",
        "- Target: Dorn mean 1-5 rating; Sergeyuk readable-vote proportion after excluding non-label choices.",
        "- Normalization: population z-score (`ddof=0`) within dataset. Pairs are within dataset and language only.",
        "- Features: 27 lexical/layout source features used by the existing RQ0 baseline, with language-aware keyword/comment handling.",
        "- Models: Linear/RF/GB/SVR regression plus canonical MLP/LR/NB/Scalabrino-LR/Voting classification (9 total).",
        "- Validation: five-fold stratified group CV; formatting-identical code hashes cannot cross train/test folds.",
        "- Pair evaluation: up to 1,000 seed-42 pairs per easy/medium/hard bin, scored from OOF predictions.",
        "",
        "## Data integrity",
        "",
        checks.to_markdown(index=False),
        "",
        f"Total snippets: {len(data)} (Dorn {int((data.dataset_name == 'Dorn').sum())}, Sergeyuk {int((data.dataset_name == 'Sergeyuk2024').sum())}).",
        f"Render audit: {int(render_audit.render_success.sum())}/{len(render_audit)} passed; failures={int((~render_audit.render_success).sum())}.",
        "Dorn mapping note: official score files have no header; their score-column counts exactly match numeric-sorted snippet counts. The older local derived CSVs were not used because their means are shifted by one snippet after an inserted 3.6667 first row.",
        "",
        "## Best snippet-level CV result per language",
        "",
        best[["dataset_name", "language", "model", "n_snippets", "spearman_r", "mae_z", "rmse_z"]].to_markdown(index=False),
        "",
        "## Pairwise OOF results",
        "",
        pair_all.to_markdown(index=False),
        "",
        "## Canonical binary CV results",
        "",
        canonical_metrics[["dataset_name", "language", "model", "n_snippets", "balanced_accuracy", "roc_auc", "macro_f1"]].to_markdown(index=False),
        "",
        "## Interpretation limits",
        "",
        "- Sergeyuk provides binary votes, not a 1-5 Likert mean; its readable-vote proportion is a different but ordered target.",
        "- OOF scores compared across folds can have fold-specific calibration shifts; Spearman and pair accuracy are screening estimates, not final held-out VLM comparisons.",
        "- Dorn's official archive/page does not state a reuse license. Keep the archive provenance and resolve redistribution permission before publishing raw files.",
        "- This stage does not run VLM inference or test swap reliability.",
        "",
        "## Reproduction",
        "",
        f"`python {Path(__file__).resolve()} --repo-root {output.parents[1]}`",
        "",
    ]
    (output / "dataset_extension_report.md").write_text("\n".join(report), encoding="utf-8")
    license_notes = """# License Notes

## Dorn dataset

- Source: https://dibt-research.unimol.it/report/readability/files/DatasetDorn.zip
- The official hosting page and downloaded archive contain no explicit license declaration.
- Provenance is recorded for research reproducibility; redistribution permission remains unresolved.

## Sergeyuk et al. 2024

- Source: https://zenodo.org/records/10550937
- DOI: https://doi.org/10.5281/zenodo.10550937
- Zenodo metadata license: CC BY 4.0.
"""
    (output / "LICENSE_NOTES.md").write_text(license_notes, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--raw-root", default="data/raw_ratings/official_20260713")
    parser.add_argument("--output", default="results/dataset_extension_20260713")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    raw_root = (repo / args.raw_root).resolve()
    output = (repo / args.output).resolve()
    for directory in [output, output / "data", output / "tables", output / "logs"]:
        directory.mkdir(parents=True, exist_ok=True)

    dorn, dorn_checks = load_dorn(raw_root)
    sergeyuk, sergeyuk_checks = load_sergeyuk(raw_root)
    data = finalize_dataset(dorn, sergeyuk)
    checks = pd.DataFrame(dorn_checks + sergeyuk_checks)
    if len(data) != 480 or data.snippet_uid.nunique() != 480:
        raise RuntimeError(f"Expected 480 unique snippets, got rows={len(data)} unique={data.snippet_uid.nunique()}")

    features = extract_features(data)
    feature_count = len([column for column in features if column.startswith("feature_")])
    if feature_count != 27 or features.isna().any().any():
        raise RuntimeError(f"Feature validation failed: count={feature_count} missing={int(features.isna().sum().sum())}")
    render_audit = render_and_audit(data, repo, output)
    if not render_audit.render_success.all():
        render_audit.to_csv(output / "tables/render_audit.csv", index=False)
        raise RuntimeError("Render audit failed; see tables/render_audit.csv")

    predictions, snippet_metrics = oof_predictions(data, features)
    canonical_predictions, canonical_metrics = oof_canonical_predictions(data, features)
    pairs, pair_availability = sampled_pairs(data)
    canonical_as_scores = canonical_predictions.rename(columns={"predicted_probability": "predicted_z"})[
        ["snippet_uid", "dataset_name", "language", "model", "fold", "predicted_z"]
    ]
    all_score_predictions = pd.concat([predictions, canonical_as_scores], ignore_index=True, sort=False)
    pair_predictions, pair_metrics = score_pairs(pairs, all_score_predictions)

    data.to_csv(output / "data/extension_snippets.csv", index=False)
    features.to_csv(output / "data/source_features_27.csv", index=False)
    predictions.to_csv(output / "data/oof_predictions.csv", index=False)
    canonical_predictions.to_csv(output / "data/oof_canonical_predictions.csv", index=False)
    pairs.to_csv(output / "data/pairs_seed42.csv", index=False)
    pair_predictions.to_csv(output / "data/pair_predictions.csv", index=False)
    checks.to_csv(output / "tables/data_integrity.csv", index=False)
    render_audit.to_csv(output / "tables/render_audit.csv", index=False)
    snippet_metrics.to_csv(output / "tables/snippet_cv_metrics.csv", index=False)
    canonical_metrics.to_csv(output / "tables/canonical_cv_metrics.csv", index=False)
    pair_availability.to_csv(output / "tables/pair_availability.csv", index=False)
    pair_metrics.to_csv(output / "tables/pair_metrics.csv", index=False)
    write_reports(output, data, checks, render_audit, snippet_metrics, canonical_metrics, pair_metrics)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "api_calls": 0,
        "raw_root": str(raw_root),
        "output": str(output),
        "snippet_count": len(data),
        "feature_count": feature_count,
        "render_passed": int(render_audit.render_success.sum()),
        "pair_count": len(pairs),
        "regression_models": 4,
        "canonical_binary_models": 5,
        "script": str(Path(__file__).resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

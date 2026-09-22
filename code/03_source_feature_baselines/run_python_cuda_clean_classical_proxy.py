#!/usr/bin/env python3
"""Re-score classical baselines and validate Dorn pair proxies on clean Python/CUDA pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon


REGRESSION_SCORE = "predicted_z"
CANONICAL_SCORE = "predicted_probability"


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan
    return float(spearmanr(x[valid].astype(float), y[valid].astype(float)).statistic)


def bh_fdr(values: pd.Series) -> pd.Series:
    p = values.astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def summarize_classical(group: pd.DataFrame) -> dict[str, object]:
    ties = group["model_tie"].astype(bool)
    correct = group["is_correct"].astype(bool)
    return {
        "pairs": int(len(group)),
        "correct_pairs": int(correct.sum()),
        "tie_pairs": int(ties.sum()),
        "pair_accuracy": float(correct.mean()),
        "spearman_continuous_all": safe_spearman(group["model_score_diff"], group["human_score_diff"]),
    }


def grouped_summary(data: pd.DataFrame, dimensions: list[str], summarizer) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in data.groupby(dimensions, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(dimensions, keys)), **summarizer(group)})
    return pd.DataFrame(rows)


def score_classical(pairs: pd.DataFrame, predictions: pd.DataFrame, score_column: str, family: str) -> pd.DataFrame:
    lookup = predictions.set_index(["language", "model", "snippet_uid"])[score_column]
    rows: list[pd.DataFrame] = []
    for (language, model), pred in predictions.groupby(["language", "model"], sort=True):
        subset = pairs[pairs["language"].eq(language)].copy()
        subset["model"] = model
        subset["model_family"] = family
        subset["model_score_i"] = [lookup.loc[(language, model, uid)] for uid in subset["snippet_i"]]
        subset["model_score_j"] = [lookup.loc[(language, model, uid)] for uid in subset["snippet_j"]]
        subset["model_score_diff"] = subset["model_score_i"] - subset["model_score_j"]
        subset["human_score_diff"] = subset["human_score_i_z"] - subset["human_score_j_z"]
        subset["model_tie"] = subset["model_score_diff"].eq(0)
        subset["model_preference"] = np.where(
            subset["model_score_diff"].gt(0), subset["snippet_i"],
            np.where(subset["model_score_diff"].lt(0), subset["snippet_j"], None),
        )
        subset["is_correct"] = subset["model_preference"].eq(subset["human_preference"])
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def load_rating_vectors(raw_root: Path, snippets: pd.DataFrame) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for language in ("python", "cuda"):
        ratings = pd.read_csv(raw_root / f"{language}.csv", header=None)
        matrix = ratings.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        language_snippets = snippets[snippets["language"].eq(language)]
        expected_ids = sorted(language_snippets["original_snippet_id"].astype(int).tolist())
        if expected_ids != list(range(matrix.shape[1])):
            raise RuntimeError(f"Unexpected {language} snippet-to-rating-column mapping")
        for row in language_snippets.itertuples(index=False):
            values = matrix.iloc[:, int(row.original_snippet_id)].astype(float).to_numpy()
            observed = values[np.isfinite(values)]
            if len(observed) != int(row.rating_count):
                raise RuntimeError(f"Rating count mismatch for {row.rq0_id}")
            if not np.isclose(observed.mean(), float(row.human_mean_score), atol=1e-12):
                raise RuntimeError(f"Rating mean mismatch for {row.rq0_id}")
            vectors[str(row.rq0_id)] = values
    return vectors


def cliff_delta_from_u(u: float, n_i: int, n_j: int) -> float:
    return float(2.0 * u / (n_i * n_j) - 1.0)


def calculate_proxy_stats(pairs: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        full_i = vectors[str(row.snippet_i)]
        full_j = vectors[str(row.snippet_j)]
        values_i = full_i[np.isfinite(full_i)]
        values_j = full_j[np.isfinite(full_j)]
        common = np.isfinite(full_i) & np.isfinite(full_j)
        common_i = full_i[common]
        common_j = full_j[common]
        test = mannwhitneyu(values_i, values_j, alternative="two-sided", method="asymptotic")
        delta = cliff_delta_from_u(float(test.statistic), len(values_i), len(values_j))
        paired_p = np.nan
        if len(common_i) >= 5 and np.any(common_i != common_j):
            paired_p = float(wilcoxon(common_i, common_j, alternative="two-sided", method="approx").pvalue)
        human_sign = np.sign(float(row.human_score_i_z) - float(row.human_score_j_z))
        delta_sign = np.sign(delta)
        rows.append(
            {
                "pair_id": row.pair_id,
                "language": row.language,
                "difficulty": row.difficulty,
                "snippet_i": row.snippet_i,
                "snippet_j": row.snippet_j,
                "rating_count_i": len(values_i),
                "rating_count_j": len(values_j),
                "common_rater_count": int(common.sum()),
                "mannwhitney_u": float(test.statistic),
                "mannwhitney_p": float(test.pvalue),
                "cliffs_delta_i_minus_j": delta,
                "paired_wilcoxon_p": paired_p,
                "paired_common_mean_diff_i_minus_j": float(np.mean(common_i - common_j)) if len(common_i) else np.nan,
                "mean_score_diff_i_minus_j": float(row.human_score_i_z) - float(row.human_score_j_z),
                "direction_matches_mean_proxy": bool(delta_sign != 0 and delta_sign == human_sign),
            }
        )
    result = pd.DataFrame(rows)
    result["mannwhitney_fdr_p"] = bh_fdr(result["mannwhitney_p"])
    paired_mask = result["paired_wilcoxon_p"].notna()
    result["paired_wilcoxon_fdr_p"] = np.nan
    result.loc[paired_mask, "paired_wilcoxon_fdr_p"] = bh_fdr(result.loc[paired_mask, "paired_wilcoxon_p"])
    result["proxy_high_confidence"] = (
        result["mannwhitney_fdr_p"].lt(0.05)
        & result["cliffs_delta_i_minus_j"].abs().ge(0.33)
        & result["direction_matches_mean_proxy"]
    )
    return result


def proxy_grid(proxy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p_limit in (0.01, 0.05, 0.10):
        for delta_limit in (0.147, 0.33, 0.474):
            selected = (
                proxy["mannwhitney_fdr_p"].lt(p_limit)
                & proxy["cliffs_delta_i_minus_j"].abs().ge(delta_limit)
                & proxy["direction_matches_mean_proxy"]
            )
            for keys, group in proxy.assign(selected=selected).groupby(["language", "difficulty"], sort=True):
                rows.append(
                    {
                        "p_threshold": p_limit,
                        "abs_delta_threshold": delta_limit,
                        "language": keys[0],
                        "difficulty": keys[1],
                        "pairs": int(len(group)),
                        "selected_pairs": int(group["selected"].sum()),
                        "selected_rate": float(group["selected"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_vlm(group: pd.DataFrame) -> dict[str, object]:
    valid = group["is_valid_strict_swap"].astype(bool)
    correct = group["is_correct"].astype(bool)
    valid_group = group[valid]
    model_sign = pd.Series(
        np.where(valid_group["model_preference"].eq(valid_group["snippet_i"]), 1.0, -1.0),
        index=valid_group.index,
    )
    human_diff = valid_group["human_score_i_z"].astype(float) - valid_group["human_score_j_z"].astype(float)
    return {
        "pairs": int(len(group)),
        "valid_pairs": int(valid.sum()),
        "correct_pairs": int(correct.sum()),
        "valid_accuracy": float(correct.sum() / valid.sum()) if valid.sum() else np.nan,
        "effective_accuracy": float(correct.mean()),
        "strict_swap_error": float((~valid).mean()),
        "spearman_valid_only": safe_spearman(model_sign, human_diff),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="results/python_cuda_missing_experiments_20260716")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output = (root / args.output).resolve()
    tables = output / "tables"
    data_dir = output / "data"
    tables.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    main_results = root / "results/python_cuda_vlm_main_20260715"
    extension = root / "results/dataset_extension_20260713/data"
    pairs = pd.read_csv(main_results / "data/pairs_seed42_clean.csv")
    snippets = pd.read_csv(main_results / "data/snippets.csv")
    regressors = pd.read_csv(extension / "oof_predictions.csv")
    canonical = pd.read_csv(extension / "oof_canonical_predictions.csv")
    regressors = regressors[regressors["language"].isin(["python", "cuda"])].copy()
    canonical = canonical[canonical["language"].isin(["python", "cuda"])].copy()

    classical = pd.concat(
        [
            score_classical(pairs, regressors, REGRESSION_SCORE, "regression"),
            score_classical(pairs, canonical, CANONICAL_SCORE, "canonical_binary_probability"),
        ],
        ignore_index=True,
    )
    if len(classical) != 9 * len(pairs):
        raise RuntimeError(f"Expected {9 * len(pairs)} classical rows, got {len(classical)}")
    classical.to_csv(data_dir / "clean_classical_pair_predictions.csv", index=False)
    for suffix, dims in {
        "overall": ["model_family", "model"],
        "by_language": ["model_family", "model", "language"],
        "by_difficulty": ["model_family", "model", "difficulty"],
        "by_language_difficulty": ["model_family", "model", "language", "difficulty"],
    }.items():
        grouped_summary(classical, dims, summarize_classical).to_csv(
            tables / f"clean_classical_results_{suffix}.csv", index=False
        )

    rating_root = root / "data/raw_ratings/official_20260713/extracted/DatasetDorn/dataset/scores"
    vectors = load_rating_vectors(rating_root, snippets)
    proxy = calculate_proxy_stats(pairs, vectors)
    proxy.to_csv(data_dir / "pair_proxy_reliability.csv", index=False)
    grouped_summary(
        proxy,
        ["language", "difficulty"],
        lambda g: {
            "pairs": int(len(g)),
            "direction_match_pairs": int(g["direction_matches_mean_proxy"].sum()),
            "direction_match_rate": float(g["direction_matches_mean_proxy"].mean()),
            "high_confidence_pairs": int(g["proxy_high_confidence"].sum()),
            "high_confidence_rate": float(g["proxy_high_confidence"].mean()),
            "median_abs_cliffs_delta": float(g["cliffs_delta_i_minus_j"].abs().median()),
            "median_mannwhitney_p": float(g["mannwhitney_p"].median()),
        },
    ).to_csv(tables / "proxy_reliability_by_language_difficulty.csv", index=False)
    proxy_grid(proxy).to_csv(tables / "proxy_threshold_grid.csv", index=False)
    paired_available = proxy[proxy["common_rater_count"].gt(0)].copy()
    paired_available["paired_direction_matches"] = (
        paired_available["paired_common_mean_diff_i_minus_j"] * paired_available["mean_score_diff_i_minus_j"]
    ).gt(0)
    grouped_summary(
        paired_available,
        ["language", "difficulty"],
        lambda g: {
            "pairs_with_common_raters": int(len(g)),
            "median_common_raters": float(g["common_rater_count"].median()),
            "paired_test_available_pairs": int(g["paired_wilcoxon_p"].notna().sum()),
            "paired_direction_match_rate": float(g["paired_direction_matches"].mean()),
            "paired_wilcoxon_fdr_significant_pairs": int(g["paired_wilcoxon_fdr_p"].lt(0.05).sum()),
        },
    ).to_csv(tables / "proxy_paired_common_rater_summary.csv", index=False)

    confidence = proxy.set_index("pair_id")["proxy_high_confidence"]
    classical_hc = classical[classical["pair_id"].map(confidence).fillna(False)].copy()
    grouped_summary(classical_hc, ["model_family", "model", "language"], summarize_classical).to_csv(
        tables / "clean_classical_results_high_confidence.csv", index=False
    )

    vlm_frames = []
    for path in sorted((main_results / "outputs").glob("*.jsonl")):
        frame = read_jsonl(path)
        frame["model"] = frame["model_name"]
        frame["setting"] = frame["input_setting"]
        frame["proxy_high_confidence"] = frame["pair_id"].map(confidence).fillna(False)
        vlm_frames.append(frame)
    vlm = pd.concat(vlm_frames, ignore_index=True)
    vlm_hc = vlm[vlm["proxy_high_confidence"]].copy()
    grouped_summary(vlm_hc, ["model", "setting"], summarize_vlm).to_csv(
        tables / "vlm_results_high_confidence_overall.csv", index=False
    )
    grouped_summary(vlm_hc, ["model", "setting", "language", "difficulty"], summarize_vlm).to_csv(
        tables / "vlm_results_high_confidence_by_language_difficulty.csv", index=False
    )

    manifest = {
        "status": "passed",
        "clean_pairs": int(len(pairs)),
        "classical_models": int(classical["model"].nunique()),
        "classical_pair_rows": int(len(classical)),
        "vlm_conditions": int(vlm.groupby(["model", "setting"]).ngroups),
        "vlm_pair_rows": int(len(vlm)),
        "proxy_high_confidence_definition": "two-sided Mann-Whitney BH-FDR p < 0.05, |Cliff's delta| >= 0.33, direction agrees with mean-score proxy",
        "proxy_high_confidence_pairs": int(proxy["proxy_high_confidence"].sum()),
        "rating_mapping_validated": True,
        "classical_oof_predictions_reused": True,
        "pair_list": str((main_results / "data/pairs_seed42_clean.csv").relative_to(root)),
    }
    (output / "classical_proxy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

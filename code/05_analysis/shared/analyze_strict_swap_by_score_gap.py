#!/usr/bin/env python3
"""Compare strict-swap errors across human-score-gap strata for clean runs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rq0_viability"
RESULTS = ROOT / "results"
OUT = RESULTS / "strict_swap_difficulty_analysis_20260710"
LEVELS = ("hard", "medium", "easy")


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def add_run(
    runs: list[pd.DataFrame],
    frame: pd.DataFrame,
    family: str,
    model: str,
    case: str,
    source: Path,
) -> None:
    required = {"pair_id", "difficulty", "is_valid_strict_swap"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing {sorted(missing)} in {source}")
    work = frame.copy()
    if work["pair_id"].duplicated().any():
        raise RuntimeError(f"Duplicate pair IDs in {source}: {case}")
    work["family"] = family
    work["model"] = model
    work["case"] = case
    work["source_file"] = str(source.relative_to(ROOT))
    work["invalid"] = (~work["is_valid_strict_swap"].astype(bool)).astype(int)
    runs.append(
        work[
            [
                "family",
                "model",
                "case",
                "source_file",
                "pair_id",
                "difficulty",
                "invalid",
            ]
        ]
    )


def load_runs() -> pd.DataFrame:
    runs: list[pd.DataFrame] = []

    primary_dir = EXP / "outputs/vlm_judges/full_factorial_clean_20260703"
    for path in sorted(primary_dir.glob("*.jsonl")):
        frame = load_jsonl(path)
        add_run(
            runs,
            frame,
            "primary_clean_full3000",
            str(frame["model_name"].iloc[0]),
            str(frame["input_setting"].iloc[0]),
            path,
        )

    openai_specs = [
        (RESULTS / "openai_vlm_pilot_20260707/gpt-5.4-mini__pilot_raw.jsonl", "gpt-5.4-mini", "none_mot24"),
        (RESULTS / "openai_vlm_pilot_20260708/gpt-5.4__pilot_raw.jsonl", "gpt-5.4", "none_mot24"),
        (
            RESULTS / "openai_vlm_pilot_20260708_gpt55_fixed/gpt-5.5__pilot_raw.jsonl",
            "gpt-5.5",
            "low_mot256",
        ),
    ]
    for path, model, setting in openai_specs:
        frame = load_jsonl(path)
        for condition, sub in frame.groupby("condition", sort=False):
            add_run(
                runs,
                sub,
                "openai_standard_300",
                model,
                f"{condition}__{setting}",
                path,
            )

    reasoning_dir = RESULTS / "gpt54_reasoning_budget_pilot_90_20260709"
    for path in sorted(reasoning_dir.glob("gpt-5.4-2026-03-05__reasoning_*__mot_*_raw.jsonl")):
        frame = load_jsonl(path)
        effort = str(frame["reasoning_effort"].iloc[0])
        budget = int(frame["max_output_tokens"].iloc[0])
        for condition, sub in frame.groupby("condition", sort=False):
            add_run(
                runs,
                sub,
                "gpt54_reasoning_budget",
                "gpt-5.4-2026-03-05",
                f"{condition}__reasoning_{effort}__mot{budget}",
                path,
            )

    max_token_dir = RESULTS / "max_token_pilot_20260709"
    for path in sorted(max_token_dir.glob("*.jsonl")):
        match = re.search(r"__mnt(\d+)\.jsonl$", path.name)
        if not match:
            raise RuntimeError(f"Cannot parse token budget from {path}")
        frame = load_jsonl(path)
        add_run(
            runs,
            frame,
            "max_token_300",
            str(frame["model_name"].iloc[0]),
            f"{frame['input_setting'].iloc[0]}__mnt{match.group(1)}",
            path,
        )

    ocr_dir = RESULTS / "screenshot_only_ocr_clean_20260707/ocr_text_llm_raw"
    ocr_specs = {
        "source_text_llm.jsonl": "source_text",
        "ocr_text_llm_easyocr.jsonl": "ocr_easyocr",
        "ocr_text_llm_rapidocr.jsonl": "ocr_rapidocr",
        "ocr_text_llm_easyocr_preprocessed.jsonl": "ocr_easyocr_preprocessed",
    }
    for filename, case in ocr_specs.items():
        path = ocr_dir / filename
        frame = load_jsonl(path)
        add_run(
            runs,
            frame,
            "ocr_text_llm_full3000",
            "Qwen2.5-Coder-7B-Instruct",
            case,
            path,
        )

    sanity_dir = EXP / "outputs/strong_vlm_sanity"
    sanity_specs = [
        (sanity_dir / "qwen7b_forced_source_raw.jsonl", "Qwen2.5-VL-7B", "forced_single_char_source"),
        (sanity_dir / "qwen32b_parse_fixed_source_raw.jsonl", "Qwen2.5-VL-32B", "single_char_source"),
    ]
    for path, model, case in sanity_specs:
        add_run(runs, load_jsonl(path), "protocol_sanity_300", model, case, path)

    data = pd.concat(runs, ignore_index=True)
    pairs = pd.read_csv(EXP / "data/pairs/full_pair_set_rq0.csv")
    pair_meta = pairs[["pair_id", "snippet_i", "snippet_j", "abs_z_diff", "difficulty"]].drop_duplicates("pair_id")
    data = data.merge(pair_meta, on="pair_id", how="left", suffixes=("", "_master"), validate="many_to_one")
    if data[["abs_z_diff", "snippet_i", "snippet_j"]].isna().any().any():
        missing = data.loc[data["abs_z_diff"].isna(), "pair_id"].drop_duplicates().head().tolist()
        raise RuntimeError(f"Pair metadata missing for {missing}")
    if not data["difficulty"].eq(data["difficulty_master"]).all():
        raise RuntimeError("Difficulty differs from the master pair set")
    data = data.drop(columns="difficulty_master")
    return data


def wilson(errors: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    p = errors / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    for _ in range(100):
        eta = np.clip(design @ beta, -30, 30)
        prob = 1 / (1 + np.exp(-eta))
        weights = np.clip(prob * (1 - prob), 1e-10, None)
        score = design.T @ (y - prob)
        hessian = design.T @ (weights[:, None] * design)
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    covariance = np.linalg.inv(hessian)
    return beta, covariance


def fixed_effect_cluster_logit(
    data: pd.DataFrame, predictors: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    run = data["model"].astype(str) + " :: " + data["case"].astype(str)
    run_dummies = pd.get_dummies(run, drop_first=True, dtype=float)
    x_frame = pd.concat(
        [pd.Series(1.0, index=data.index, name="intercept"), predictors, run_dummies], axis=1
    )
    x = x_frame.to_numpy(dtype=float)
    y = data["invalid"].to_numpy(dtype=float)
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        eta = np.clip(x @ beta, -30, 30)
        prob = 1 / (1 + np.exp(-eta))
        weights = np.clip(prob * (1 - prob), 1e-10, None)
        score = x.T @ (y - prob)
        hessian = x.T @ (weights[:, None] * x)
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    bread = np.linalg.inv(hessian)
    score_rows = x * (y - prob)[:, None]
    score_frame = pd.DataFrame(score_rows)
    score_frame["pair_id"] = data["pair_id"].to_numpy()
    cluster_scores = score_frame.groupby("pair_id", sort=False).sum().drop(columns="pair_id", errors="ignore").to_numpy()
    meat = cluster_scores.T @ cluster_scores
    covariance = bread @ meat @ bread
    clusters = data["pair_id"].nunique()
    observations = len(data)
    parameters = x.shape[1]
    covariance *= (clusters / (clusters - 1)) * ((observations - 1) / (observations - parameters))
    return beta, covariance, x_frame.columns.tolist()


def holm(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    order = valid.sort_values().index
    running = 0.0
    m = len(valid)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, float(valid.loc[idx]) * (m - rank)))
        result.loc[idx] = running
    return result


def snippet_cluster_bootstrap(
    data: pd.DataFrame, reps: int = 10_000, seed: int = 20260710
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = data[data["family"].eq("primary_clean_full3000")]
    pairs = (
        primary.groupby(
            ["pair_id", "snippet_i", "snippet_j", "difficulty", "abs_z_diff"], as_index=False
        )["invalid"]
        .mean()
        .rename(columns={"invalid": "condition_mean_swap"})
    )
    snippets = sorted(set(pairs["snippet_i"]) | set(pairs["snippet_j"]))
    lookup = {snippet: idx for idx, snippet in enumerate(snippets)}
    idx_i = pairs["snippet_i"].map(lookup).to_numpy(dtype=int)
    idx_j = pairs["snippet_j"].map(lookup).to_numpy(dtype=int)
    gap = pairs["abs_z_diff"].to_numpy(dtype=float)
    outcome = pairs["condition_mean_swap"].to_numpy(dtype=float)
    easy = pairs["difficulty"].eq("easy").to_numpy()
    hard = pairs["difficulty"].eq("hard").to_numpy()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    probabilities = np.full(len(snippets), 1 / len(snippets))
    for rep in range(reps):
        counts = rng.multinomial(len(snippets), probabilities)
        weights = counts[idx_i] * counts[idx_j]
        easy_weight = weights[easy].sum()
        hard_weight = weights[hard].sum()
        if easy_weight == 0 or hard_weight == 0 or weights.sum() == 0:
            continue
        easy_rate = float(np.average(outcome[easy], weights=weights[easy]))
        hard_rate = float(np.average(outcome[hard], weights=weights[hard]))
        mean_gap = float(np.average(gap, weights=weights))
        mean_outcome = float(np.average(outcome, weights=weights))
        denominator = float(np.sum(weights * (gap - mean_gap) ** 2))
        slope = (
            float(np.sum(weights * (gap - mean_gap) * (outcome - mean_outcome)) / denominator)
            if denominator
            else np.nan
        )
        rows.append(
            {
                "replicate": rep,
                "easy_swap": easy_rate,
                "hard_swap": hard_rate,
                "easy_minus_hard": easy_rate - hard_rate,
                "weighted_linear_slope_per_abs_z": slope,
            }
        )
    bootstrap = pd.DataFrame(rows)
    point_easy = float(outcome[easy].mean())
    point_hard = float(outcome[hard].mean())
    point_slope = float(np.cov(gap, outcome, ddof=0)[0, 1] / np.var(gap))
    summary = pd.DataFrame(
        [
            {
                "estimand": "easy_minus_hard_swap",
                "point_estimate": point_easy - point_hard,
                "bootstrap_mean": bootstrap["easy_minus_hard"].mean(),
                "ci_low": bootstrap["easy_minus_hard"].quantile(0.025),
                "ci_high": bootstrap["easy_minus_hard"].quantile(0.975),
                "replicates": len(bootstrap),
            },
            {
                "estimand": "linear_probability_slope_per_abs_z",
                "point_estimate": point_slope,
                "bootstrap_mean": bootstrap["weighted_linear_slope_per_abs_z"].mean(),
                "ci_low": bootstrap["weighted_linear_slope_per_abs_z"].quantile(0.025),
                "ci_high": bootstrap["weighted_linear_slope_per_abs_z"].quantile(0.975),
                "replicates": len(bootstrap),
            },
        ]
    )
    return bootstrap, summary


def analyze(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    keys = ["family", "model", "case", "source_file"]
    for key, group in data.groupby(keys, sort=False):
        family, model, case, source_file = key
        counts: dict[str, tuple[int, int, float]] = {}
        for level in LEVELS:
            sub = group[group["difficulty"].eq(level)]
            n = len(sub)
            errors = int(sub["invalid"].sum())
            rate = errors / n
            ci_low, ci_high = wilson(errors, n)
            counts[level] = (n, errors, rate)
            long_rows.append(
                {
                    "family": family,
                    "model": model,
                    "case": case,
                    "difficulty": level,
                    "pairs": n,
                    "swap_errors": errors,
                    "strict_swap_error": rate,
                    "wilson_ci_low": ci_low,
                    "wilson_ci_high": ci_high,
                    "score_gap_min": float(sub["abs_z_diff"].min()),
                    "score_gap_max": float(sub["abs_z_diff"].max()),
                    "score_gap_mean": float(sub["abs_z_diff"].mean()),
                }
            )
        table = np.array(
            [[counts[level][1], counts[level][0] - counts[level][1]] for level in LEVELS]
        )
        chi2, chi_p, _, _ = chi2_contingency(table)
        beta, covariance = logistic_fit(
            group["abs_z_diff"].to_numpy(dtype=float), group["invalid"].to_numpy(dtype=float)
        )
        slope = float(beta[1])
        slope_se = float(math.sqrt(covariance[1, 1]))
        z_value = slope / slope_se
        slope_p = float(2 * norm.sf(abs(z_value)))
        odds_ratio = math.exp(slope)
        or_low = math.exp(slope - 1.959963984540054 * slope_se)
        or_high = math.exp(slope + 1.959963984540054 * slope_se)
        hard_rate = counts["hard"][2]
        medium_rate = counts["medium"][2]
        easy_rate = counts["easy"][2]
        common = {
            "family": family,
            "model": model,
            "case": case,
            "pairs": len(group),
            "source_file": source_file,
        }
        wide_rows.append(
            {
                **common,
                "hard_swap_error": hard_rate,
                "medium_swap_error": medium_rate,
                "easy_swap_error": easy_rate,
                "easy_minus_hard": easy_rate - hard_rate,
                "expected_monotonic_hard_gt_medium_gt_easy": bool(
                    hard_rate > medium_rate > easy_rate
                ),
            }
        )
        association_rows.append(
            {
                **common,
                "chi_square": float(chi2),
                "difficulty_heterogeneity_p": float(chi_p),
                "logit_slope_per_abs_z": slope,
                "logit_slope_se": slope_se,
                "odds_ratio_per_plus1_abs_z": odds_ratio,
                "odds_ratio_ci_low": or_low,
                "odds_ratio_ci_high": or_high,
                "continuous_gap_p": slope_p,
            }
        )
    wide = pd.DataFrame(wide_rows)
    association = pd.DataFrame(association_rows)
    association["continuous_gap_holm_all50"] = holm(association["continuous_gap_p"])
    primary_mask = association["family"].eq("primary_clean_full3000")
    association.loc[primary_mask, "continuous_gap_holm_primary8"] = holm(
        association.loc[primary_mask, "continuous_gap_p"]
    )

    primary = data[data["family"].eq("primary_clean_full3000")].reset_index(drop=True)
    pooled_rows: list[dict[str, object]] = []
    beta, covariance, names = fixed_effect_cluster_logit(
        primary, pd.DataFrame({"abs_z_diff": primary["abs_z_diff"].to_numpy()}, index=primary.index)
    )
    idx = names.index("abs_z_diff")
    se = math.sqrt(covariance[idx, idx])
    pooled_rows.append(
        {
            "analysis": "continuous_gap_condition_FE_pair_clustered",
            "contrast": "+1 abs human z gap",
            "coefficient": beta[idx],
            "cluster_robust_se": se,
            "odds_ratio": math.exp(beta[idx]),
            "or_ci_low": math.exp(beta[idx] - 1.959963984540054 * se),
            "or_ci_high": math.exp(beta[idx] + 1.959963984540054 * se),
            "p_value": 2 * norm.sf(abs(beta[idx] / se)),
            "observations": len(primary),
            "pair_clusters": primary["pair_id"].nunique(),
        }
    )
    categorical = pd.DataFrame(
        {
            "medium_vs_hard": primary["difficulty"].eq("medium").astype(float),
            "easy_vs_hard": primary["difficulty"].eq("easy").astype(float),
        },
        index=primary.index,
    )
    beta, covariance, names = fixed_effect_cluster_logit(primary, categorical)
    for contrast in ("medium_vs_hard", "easy_vs_hard"):
        idx = names.index(contrast)
        se = math.sqrt(covariance[idx, idx])
        pooled_rows.append(
            {
                "analysis": "difficulty_condition_FE_pair_clustered",
                "contrast": contrast,
                "coefficient": beta[idx],
                "cluster_robust_se": se,
                "odds_ratio": math.exp(beta[idx]),
                "or_ci_low": math.exp(beta[idx] - 1.959963984540054 * se),
                "or_ci_high": math.exp(beta[idx] + 1.959963984540054 * se),
                "p_value": 2 * norm.sf(abs(beta[idx] / se)),
                "observations": len(primary),
                "pair_clusters": primary["pair_id"].nunique(),
            }
        )
    for level in LEVELS:
        sub = primary[primary["difficulty"].eq(level)].reset_index(drop=True)
        beta, covariance, names = fixed_effect_cluster_logit(
            sub, pd.DataFrame({"abs_z_diff": sub["abs_z_diff"].to_numpy()}, index=sub.index)
        )
        idx = names.index("abs_z_diff")
        se = math.sqrt(covariance[idx, idx])
        pooled_rows.append(
            {
                "analysis": "within_difficulty_continuous_condition_FE_pair_clustered",
                "contrast": f"+1 abs human z gap within {level}",
                "coefficient": beta[idx],
                "cluster_robust_se": se,
                "odds_ratio": math.exp(beta[idx]),
                "or_ci_low": math.exp(beta[idx] - 1.959963984540054 * se),
                "or_ci_high": math.exp(beta[idx] + 1.959963984540054 * se),
                "p_value": 2 * norm.sf(abs(beta[idx] / se)),
                "observations": len(sub),
                "pair_clusters": sub["pair_id"].nunique(),
            }
        )

    unique_pairs = primary[["pair_id", "abs_z_diff"]].drop_duplicates("pair_id").copy()
    unique_pairs["gap_decile"] = pd.qcut(
        unique_pairs["abs_z_diff"], 10, labels=False, duplicates="drop"
    ) + 1
    decile_data = primary.merge(unique_pairs[["pair_id", "gap_decile"]], on="pair_id", validate="many_to_one")
    deciles = (
        decile_data.groupby("gap_decile")
        .agg(
            unique_pairs=("pair_id", "nunique"),
            pair_run_rows=("invalid", "size"),
            score_gap_min=("abs_z_diff", "min"),
            score_gap_max=("abs_z_diff", "max"),
            score_gap_mean=("abs_z_diff", "mean"),
            strict_swap_error=("invalid", "mean"),
        )
        .reset_index()
    )
    easy_pairs = unique_pairs[unique_pairs["abs_z_diff"].ge(1.0)].copy()
    easy_pairs["easy_gap_quartile"] = pd.qcut(easy_pairs["abs_z_diff"], 4, labels=False) + 1
    easy_data = primary[primary["difficulty"].eq("easy")].merge(
        easy_pairs[["pair_id", "easy_gap_quartile"]], on="pair_id", validate="many_to_one"
    )
    easy_quartiles = (
        easy_data.groupby("easy_gap_quartile")
        .agg(
            unique_pairs=("pair_id", "nunique"),
            pair_run_rows=("invalid", "size"),
            score_gap_min=("abs_z_diff", "min"),
            score_gap_max=("abs_z_diff", "max"),
            score_gap_mean=("abs_z_diff", "mean"),
            strict_swap_error=("invalid", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(long_rows), wide, association, pd.DataFrame(pooled_rows), deciles, easy_quartiles


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_runs()
    long, wide, association, pooled, deciles, easy_quartiles = analyze(data)
    snippet_bootstrap, snippet_bootstrap_summary = snippet_cluster_bootstrap(data)
    expected_cases = {
        "primary_clean_full3000": 8,
        "openai_standard_300": 12,
        "gpt54_reasoning_budget": 8,
        "max_token_300": 16,
        "ocr_text_llm_full3000": 4,
        "protocol_sanity_300": 2,
    }
    actual_cases = wide.groupby("family").size().to_dict()
    if actual_cases != expected_cases:
        raise RuntimeError(f"Unexpected run inventory: {actual_cases}")
    long.to_csv(OUT / "all_cases_difficulty_long.csv", index=False)
    wide.to_csv(OUT / "all_cases_difficulty_wide.csv", index=False)
    association.to_csv(OUT / "all_cases_gap_association.csv", index=False)
    pooled.to_csv(OUT / "primary_pooled_gap_models.csv", index=False)
    deciles.to_csv(OUT / "primary_gap_deciles.csv", index=False)
    easy_quartiles.to_csv(OUT / "primary_easy_gap_quartiles.csv", index=False)
    snippet_bootstrap.to_csv(OUT / "primary_snippet_cluster_bootstrap_replicates.csv", index=False)
    snippet_bootstrap_summary.to_csv(OUT / "primary_snippet_cluster_bootstrap_summary.csv", index=False)
    manifest = {
        "cases": int(len(wide)),
        "pair_run_rows": int(len(data)),
        "families": actual_cases,
        "difficulty_definition": {
            "easy": "abs human z gap >= 1.0",
            "medium": "0.5 <= abs human z gap < 1.0",
            "hard": "0.2 <= abs human z gap < 0.5",
        },
        "excluded": [
            "pre-clean/stale/corrupted-render runs",
            "smoke and duplicate result-tree copies",
            "failed partial GPT-5.5 run (422 rows); fixed complete run used",
            "presentation perturbation cases without score-gap difficulty strata",
            "classical ML swap error, structurally zero by score-difference construction",
        ],
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    primary_wide = wide[wide["family"].eq("primary_clean_full3000")]
    family_summary = (
        wide.groupby("family")
        .agg(
            cases=("case", "size"),
            mean_hard_swap=("hard_swap_error", "mean"),
            mean_medium_swap=("medium_swap_error", "mean"),
            mean_easy_swap=("easy_swap_error", "mean"),
            hard_greater_easy_cases=("easy_minus_hard", lambda x: int((x < 0).sum())),
            expected_monotonic_cases=("expected_monotonic_hard_gt_medium_gt_easy", "sum"),
        )
        .reset_index()
    )
    family_summary.to_csv(OUT / "family_summary.csv", index=False)
    (OUT / "analysis.md").write_text(
        "# Strict-swap error by human-score gap\n\n"
        "## Inventory\n\n"
        + pd.DataFrame([manifest]).to_markdown(index=False)
        + "\n\n## Primary clean 3,000-pair cases\n\n"
        + primary_wide.to_markdown(index=False, floatfmt=".6f")
        + "\n\n## Family summary\n\n"
        + family_summary.to_markdown(index=False, floatfmt=".6f")
        + "\n\n## Primary pooled models\n\n"
        + pooled.to_markdown(index=False, floatfmt=".6f")
        + "\n\n## Primary gap deciles\n\n"
        + deciles.to_markdown(index=False, floatfmt=".6f")
        + "\n\n## Easy-stratum gap quartiles\n\n"
        + easy_quartiles.to_markdown(index=False, floatfmt=".6f")
        + "\n\n## Snippet-cluster bootstrap\n\n"
        + snippet_bootstrap_summary.to_markdown(index=False, floatfmt=".6f")
        + "\n",
        encoding="utf-8",
    )
    print(wide.to_string(index=False))
    print("\nPRIMARY POOLED\n", pooled.to_string(index=False))
    print("\nFAMILY SUMMARY\n", family_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

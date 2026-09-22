#!/usr/bin/env python3
"""Run FSE 2027 reinforcement analyses A-D without new model inference."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
GRID_ROOT = REPO / "results/grounded_protocol_3lang_20260721"
BATTERY_ROOT = REPO / "results/rq1_model_battery_3lang_20260723"
GRID_PAIR_PATH = GRID_ROOT / "analysis/logit_decomposition/LOGIT_PAIR_LEVEL.csv"
BATTERY_PAIR_PATH = BATTERY_ROOT / "analysis/full/pair_level_results.csv"
BATTERY_META_PATH = BATTERY_ROOT / "data/rq1_pairs_9000.csv"
BOOTSTRAP_REPS = 10_000
SEED = 20260729

MODEL_LABELS = {
    "qwen": "Qwen2.5-VL-7B",
    "internvl": "InternVL3-8B",
    "gemma": "Gemma-3-12B",
    "ministral": "Ministral-3-8B",
    "phi": "Phi-4-MM",
}


def require_ga0() -> None:
    path = OUT / "audit/GA0_REPORT.json"
    if not path.exists() or json.loads(path.read_text()).get("status") != "passed":
        raise RuntimeError("GA0 has not passed")


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def exact_interval(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    interval = binomtest(successes, n).proportion_ci(confidence_level=0.95, method="exact")
    return float(interval.low), float(interval.high)


def parse_condition(condition: str) -> tuple[str, str, str]:
    parts = condition.split("__")
    if len(parts) != 4 or not parts[1].startswith("fs") or not parts[2].startswith("wrap"):
        raise ValueError(f"Unexpected grid condition: {condition}")
    return parts[0], parts[1], parts[2]


def factor_class(condition_a: str, condition_b: str) -> str:
    left = parse_condition(condition_a)
    right = parse_condition(condition_b)
    changed = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
    if changed == [0]:
        return "theme_only"
    if changed == [1]:
        return "font_only"
    if changed == [2]:
        return "wrap_only"
    return "compound"


def analysis_a(grid: pd.DataFrame) -> dict:
    target = OUT / "A_preference_flip"
    target.mkdir(parents=True, exist_ok=True)
    grid = grid[grid["experiment"].eq("grid")].copy()
    expected = 2 * 12 * 3 * 1000
    if len(grid) != expected:
        raise ValueError(f"Grid pair row drift: {len(grid)} != {expected}")

    rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    for (model, language), group in grid.groupby(["model", "language"], sort=True):
        conditions = sorted(group["condition"].unique())
        if len(conditions) != 12:
            raise ValueError(f"{model}/{language}: expected 12 conditions")
        by_condition = {
            condition: frame.set_index("pair_id").sort_index()
            for condition, frame in group.groupby("condition", sort=True)
        }
        pair_index = by_condition[conditions[0]].index
        for condition, frame in by_condition.items():
            if not frame.index.equals(pair_index):
                raise ValueError(f"{model}/{language}/{condition}: pair coverage drift")
            swap_count = int((~frame["strict_valid"]).sum())
            low, high = exact_interval(swap_count, len(frame))
            baseline_rows.append(
                {
                    "run": "grounded_protocol_3lang_20260721",
                    "model": model,
                    "language": language,
                    "condition": condition,
                    "n_pairs": len(frame),
                    "strict_swap_pairs": swap_count,
                    "strict_swap_error": swap_count / len(frame),
                    "strict_swap_ci_low": low,
                    "strict_swap_ci_high": high,
                    "repeat_run_available": False,
                    "repeat_run_flip_rate": math.nan,
                }
            )
        for condition_a, condition_b in combinations(conditions, 2):
            left = by_condition[condition_a]
            right = by_condition[condition_b]
            strict_mask = left["strict_valid"] & right["strict_valid"]
            for order, column in [("AB", "selected_ab"), ("BA", "selected_ba")]:
                flips = left[column].ne(right[column])
                count = int(flips.sum())
                low, high = exact_interval(count, len(flips))
                p_greater = float(binomtest(count, len(flips), 0.5, alternative="greater").pvalue)
                rows.append(
                    {
                        "run": "grounded_protocol_3lang_20260721",
                        "model": model,
                        "language": language,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "factor_class": factor_class(condition_a, condition_b),
                        "metric": f"fixed_order_{order}",
                        "n_pairs": len(flips),
                        "flip_pairs": count,
                        "flip_rate": count / len(flips),
                        "ci_low": low,
                        "ci_high": high,
                        "exact_p_greater_than_0_5": p_greater,
                    }
                )
            strict_flips = left.loc[strict_mask, "selected_ab"].ne(
                right.loc[strict_mask, "selected_ab"]
            )
            strict_count = int(strict_flips.sum())
            low, high = exact_interval(strict_count, len(strict_flips))
            p_greater = (
                float(binomtest(strict_count, len(strict_flips), 0.5, alternative="greater").pvalue)
                if len(strict_flips)
                else math.nan
            )
            rows.append(
                {
                    "run": "grounded_protocol_3lang_20260721",
                    "model": model,
                    "language": language,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "factor_class": factor_class(condition_a, condition_b),
                    "metric": "strict_valid_both",
                    "n_pairs": len(strict_flips),
                    "flip_pairs": strict_count,
                    "flip_rate": strict_count / len(strict_flips) if len(strict_flips) else math.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "exact_p_greater_than_0_5": p_greater,
                }
            )

    matrix = pd.DataFrame(rows)
    matrix["holm_p_within_model_language_metric"] = math.nan
    for _, indices in matrix.groupby(["model", "language", "metric"]).groups.items():
        idx = list(indices)
        matrix.loc[idx, "holm_p_within_model_language_metric"] = holm_adjust(
            matrix.loc[idx, "exact_p_greater_than_0_5"].fillna(1.0).tolist()
        )
    matrix["holm_reject_greater_than_0_5"] = (
        matrix["holm_p_within_model_language_metric"] < 0.05
    )
    matrix.to_csv(target / "A_flip_matrix.csv", index=False)

    summaries: list[dict[str, object]] = []
    for keys, group in matrix.groupby(["model", "language", "metric"], sort=True):
        maximum = group.loc[group["flip_rate"].idxmax()]
        summaries.append(
            {
                "model": keys[0],
                "language": keys[1],
                "metric": keys[2],
                "condition_pairs": len(group),
                "median_flip_rate": group["flip_rate"].median(),
                "max_flip_rate": maximum["flip_rate"],
                "max_n_pairs": int(maximum["n_pairs"]),
                "max_condition_a": maximum["condition_a"],
                "max_condition_b": maximum["condition_b"],
                "holm_significant_greater_than_0_5": int(
                    group["holm_reject_greater_than_0_5"].sum()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(target / "A_flip_summary.csv", index=False)

    factor = (
        matrix.groupby(["model", "language", "metric", "factor_class"], sort=True)
        .agg(
            condition_pairs=("flip_rate", "size"),
            total_denominator=("n_pairs", "sum"),
            mean_flip_rate=("flip_rate", "mean"),
            median_flip_rate=("flip_rate", "median"),
            max_flip_rate=("flip_rate", "max"),
        )
        .reset_index()
    )
    factor.to_csv(target / "A_flip_by_factor.csv", index=False)

    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(target / "A_reference_baselines.csv", index=False)
    baseline_summary = (
        baseline.groupby(["model", "language"], sort=True)
        .agg(
            conditions=("condition", "size"),
            median_strict_swap_error=("strict_swap_error", "median"),
            mean_strict_swap_error=("strict_swap_error", "mean"),
            min_strict_swap_error=("strict_swap_error", "min"),
            max_strict_swap_error=("strict_swap_error", "max"),
        )
        .reset_index()
    )
    baseline_summary["repeat_run_status"] = (
        "not measurable: no same-condition repeated full run; no new calls made"
    )
    baseline_summary.to_csv(target / "A_reference_baseline_summary.csv", index=False)

    compact = summary.pivot_table(
        index=["model", "language"],
        columns="metric",
        values=["median_flip_rate", "max_flip_rate"],
    )
    compact.columns = ["_".join(column) for column in compact.columns]
    compact = compact.reset_index().merge(
        baseline_summary[
            ["model", "language", "median_strict_swap_error", "mean_strict_swap_error"]
        ],
        on=["model", "language"],
    )
    compact.to_csv(target / "A_paper_summary_table.csv", index=False)
    return {
        "matrix_rows": len(matrix),
        "condition_pairs_per_model_language": 66,
        "repeat_run_baseline": "not measurable",
        "paper_table": compact,
    }


def deterministic_snippet_split(snippets: pd.DataFrame) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for language, values in snippets.groupby("language", sort=True):
        unique = sorted(set(values["snippet_key"]))
        ordered = sorted(
            unique,
            key=lambda item: hashlib.sha256(f"{SEED}|{item}".encode()).hexdigest(),
        )
        midpoint = len(ordered) // 2
        for item in ordered[:midpoint]:
            assignments[item] = "train"
        for item in ordered[midpoint:]:
            assignments[item] = "test"
    return assignments


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10
) -> float:
    order = np.argsort(probabilities, kind="mergesort")
    chunks = np.array_split(order, bins)
    total = len(probabilities)
    return float(
        sum(
            len(chunk) / total
            * abs(float(probabilities[chunk].mean()) - float(outcomes[chunk].mean()))
            for chunk in chunks
            if len(chunk)
        )
    )


def assign_deciles(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    non_tie = group[~group["content_tie"]].sort_values(["abs_c", "pair_id"])
    n = len(non_tie)
    positions = np.arange(n)
    deciles = np.floor(positions * 10 / n).astype(int) + 1
    group["decile"] = 1
    group.loc[non_tie.index, "decile"] = deciles
    return group


def two_way_snippet_bootstrap_difference(
    group: pd.DataFrame,
    *,
    outcome: str,
    denominator_mask: str | None,
    reps: int,
    seed: int,
) -> tuple[float, float, float, int]:
    selected = group[group["decile"].isin([1, 10])].copy()
    snippets = sorted(set(selected["snippet_i_key"]) | set(selected["snippet_j_key"]))
    lookup = {snippet: index for index, snippet in enumerate(snippets)}
    i_index = selected["snippet_i_key"].map(lookup).to_numpy()
    j_index = selected["snippet_j_key"].map(lookup).to_numpy()
    decile = selected["decile"].to_numpy()
    y = selected[outcome].astype(float).to_numpy()
    denom = (
        selected[denominator_mask].astype(float).to_numpy()
        if denominator_mask
        else np.ones(len(selected), dtype=float)
    )
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(reps):
        multiplicity = rng.multinomial(len(snippets), np.full(len(snippets), 1 / len(snippets)))
        weights = multiplicity[i_index] * multiplicity[j_index]
        values = []
        for target_decile in (1, 10):
            mask = decile == target_decile
            denominator = np.sum(weights[mask] * denom[mask])
            numerator = np.sum(weights[mask] * y[mask])
            values.append(numerator / denominator if denominator > 0 else math.nan)
        if np.isfinite(values).all():
            differences.append(values[1] - values[0])
    if not differences:
        return math.nan, math.nan, math.nan, 0
    observed_values = []
    for target_decile in (1, 10):
        mask = decile == target_decile
        denominator = float(denom[mask].sum())
        observed_values.append(float(y[mask].sum() / denominator))
    array = np.asarray(differences)
    return (
        observed_values[1] - observed_values[0],
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        len(array),
    )


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    batch = 250
    cursor = 0
    while cursor < reps:
        count = min(batch, reps - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def analysis_b(battery: pd.DataFrame) -> dict:
    target = OUT / "B_content_calibration"
    target.mkdir(parents=True, exist_ok=True)
    battery = pd.concat(
        [assign_deciles(group) for _, group in battery.groupby("model_key", sort=False)],
        ignore_index=True,
    )
    if set(battery["decile"]) != set(range(1, 11)):
        raise ValueError("Failed to construct all ten deciles")
    battery.to_csv(target / "B_pair_decile_assignments.csv", index=False)

    decile_rows: list[dict[str, object]] = []
    scopes = [("overall", "all", battery)]
    scopes.extend(
        ("language", language, frame)
        for language, frame in battery.groupby("language", sort=True)
    )
    for scope, scope_value, scope_frame in scopes:
        for (model_key, decile), group in scope_frame.groupby(
            ["model_key", "decile"], sort=True
        ):
            valid = group["valid"]
            correct = group["correct"]
            non_tie = ~group["content_tie"]
            valid_non_tie = valid & non_tie
            decile_rows.append(
                {
                    "run": "rq1_model_battery_3lang_20260723",
                    "model_key": model_key,
                    "model": group.iloc[0]["model"],
                    "scope": scope,
                    "language": scope_value,
                    "decile": decile,
                    "n_pairs": len(group),
                    "c_ties": int(group["content_tie"].sum()),
                    "strict_valid_pairs": int(valid.sum()),
                    "correct_strict_valid_pairs": int(correct.sum()),
                    "valid_accuracy": correct.sum() / valid.sum() if valid.sum() else math.nan,
                    "effective_accuracy": correct.mean(),
                    "debiased_accuracy": group["debiased_correct"].mean(),
                    "non_tie_pairs": int(non_tie.sum()),
                    "valid_accuracy_non_tie": (
                        correct[non_tie].sum() / valid_non_tie.sum()
                        if valid_non_tie.sum()
                        else math.nan
                    ),
                    "effective_accuracy_non_tie": correct[non_tie].mean(),
                    "debiased_accuracy_non_tie": group.loc[
                        non_tie, "debiased_correct"
                    ].mean(),
                    "abs_c_min": group["abs_c"].min(),
                    "abs_c_median": group["abs_c"].median(),
                    "abs_c_max": group["abs_c"].max(),
                }
            )
    deciles = pd.DataFrame(decile_rows)
    complete_index = pd.MultiIndex.from_product(
        [
            sorted(battery["model_key"].unique()),
            [("overall", "all"), ("language", "cuda"), ("language", "java"), ("language", "python")],
            range(1, 11),
        ],
        names=["model_key", "scope_language", "decile"],
    )
    complete_rows = pd.DataFrame(
        [
            {
                "model_key": model_key,
                "scope": scope_language[0],
                "language": scope_language[1],
                "decile": decile,
            }
            for model_key, scope_language, decile in complete_index
        ]
    )
    deciles = complete_rows.merge(
        deciles,
        on=["model_key", "scope", "language", "decile"],
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "n_pairs",
        "c_ties",
        "strict_valid_pairs",
        "correct_strict_valid_pairs",
        "non_tie_pairs",
    ]
    deciles[count_columns] = deciles[count_columns].fillna(0).astype(int)
    deciles["run"] = deciles["run"].fillna("rq1_model_battery_3lang_20260723")
    deciles["model"] = deciles["model"].fillna(
        deciles["model_key"].map(
            battery.drop_duplicates("model_key").set_index("model_key")["model"]
        )
    )
    deciles.to_csv(target / "B_calibration_by_decile.csv", index=False)

    assignments = deterministic_snippet_split(
        pd.concat(
            [
                battery[["language", "snippet_i_key"]].rename(
                    columns={"snippet_i_key": "snippet_key"}
                ),
                battery[["language", "snippet_j_key"]].rename(
                    columns={"snippet_j_key": "snippet_key"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates()
    )
    split_table = pd.DataFrame(
        [{"snippet_key": key, "split": value} for key, value in assignments.items()]
    )
    split_table.to_csv(target / "B_snippet_disjoint_split.csv", index=False)
    battery["split_i"] = battery["snippet_i_key"].map(assignments)
    battery["split_j"] = battery["snippet_j_key"].map(assignments)
    battery["calibration_role"] = np.where(
        (battery["split_i"] == "train") & (battery["split_j"] == "train"),
        "train",
        np.where(
            (battery["split_i"] == "test") & (battery["split_j"] == "test"),
            "test",
            "cross_excluded",
        ),
    )

    summary_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []
    for model_index, (model_key, model_frame) in enumerate(
        battery.groupby("model_key", sort=True)
    ):
        model_scopes = [("overall", "all", model_frame)]
        model_scopes.extend(
            ("language", language, frame)
            for language, frame in model_frame.groupby("language", sort=True)
        )
        for scope_index, (scope, language, group) in enumerate(model_scopes):
            dec = deciles[
                deciles["model_key"].eq(model_key)
                & deciles["scope"].eq(scope)
                & deciles["language"].eq(language)
            ].sort_values("decile")
            dec_nonempty = dec[dec["n_pairs"].gt(0)]
            def decile_rho(column: str) -> float:
                available = dec_nonempty.dropna(subset=[column])
                return (
                    float(spearmanr(available["decile"], available[column]).statistic)
                    if len(available) >= 2
                    else math.nan
                )

            rho_valid = decile_rho("valid_accuracy")
            rho_effective = decile_rho("effective_accuracy")
            rho_debiased = decile_rho("debiased_accuracy")
            rho_valid_non_tie = decile_rho("valid_accuracy_non_tie")
            rho_effective_non_tie = decile_rho("effective_accuracy_non_tie")
            rho_debiased_non_tie = decile_rho("debiased_accuracy_non_tie")
            seed_base = SEED + model_index * 100 + scope_index * 10
            eff_diff, eff_low, eff_high, eff_reps = two_way_snippet_bootstrap_difference(
                group,
                outcome="correct",
                denominator_mask=None,
                reps=BOOTSTRAP_REPS,
                seed=seed_base,
            )
            valid_diff, valid_low, valid_high, valid_reps = (
                two_way_snippet_bootstrap_difference(
                    group,
                    outcome="correct",
                    denominator_mask="valid",
                    reps=BOOTSTRAP_REPS,
                    seed=seed_base + 1,
                )
            )
            deb_diff, deb_low, deb_high, deb_reps = (
                two_way_snippet_bootstrap_difference(
                    group,
                    outcome="debiased_correct",
                    denominator_mask=None,
                    reps=BOOTSTRAP_REPS,
                    seed=seed_base + 2,
                )
            )
            non_tie_group = group[~group["content_tie"]]
            eff_nt_diff, eff_nt_low, eff_nt_high, eff_nt_reps = (
                two_way_snippet_bootstrap_difference(
                    non_tie_group,
                    outcome="correct",
                    denominator_mask=None,
                    reps=BOOTSTRAP_REPS,
                    seed=seed_base + 3,
                )
            )
            deb_nt_diff, deb_nt_low, deb_nt_high, deb_nt_reps = (
                two_way_snippet_bootstrap_difference(
                    non_tie_group,
                    outcome="debiased_correct",
                    denominator_mask=None,
                    reps=BOOTSTRAP_REPS,
                    seed=seed_base + 4,
                )
            )

            train = group[group["calibration_role"].eq("train")]
            test = group[group["calibration_role"].eq("test")]
            cross = group[group["calibration_role"].eq("cross_excluded")]
            if not set(train["snippet_i_key"]).isdisjoint(set(test["snippet_i_key"])):
                raise ValueError("Calibration split leaks snippet_i")
            train_snippets = set(train["snippet_i_key"]) | set(train["snippet_j_key"])
            test_snippets = set(test["snippet_i_key"]) | set(test["snippet_j_key"])
            if not train_snippets.isdisjoint(test_snippets):
                raise ValueError("Calibration split has snippet leakage")
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(train["abs_c"].to_numpy(), train["debiased_correct"].astype(int).to_numpy())
            calibrated = iso.predict(test["abs_c"].to_numpy())
            outcomes = test["debiased_correct"].astype(int).to_numpy()
            ece = expected_calibration_error(calibrated, outcomes)
            auroc = (
                float(roc_auc_score(outcomes, calibrated))
                if len(np.unique(outcomes)) == 2
                else math.nan
            )
            raw_auroc = (
                float(roc_auc_score(outcomes, test["abs_c"]))
                if len(np.unique(outcomes)) == 2
                else math.nan
            )
            summary_rows.append(
                {
                    "run": "rq1_model_battery_3lang_20260723",
                    "model_key": model_key,
                    "model": group.iloc[0]["model"],
                    "scope": scope,
                    "language": language,
                    "n_pairs": len(group),
                    "bootstrap_reps_requested": BOOTSTRAP_REPS,
                    "c_ties_conservative_failures": int(group["content_tie"].sum()),
                    "rho_decile_valid_accuracy": rho_valid,
                    "rho_decile_effective_accuracy": rho_effective,
                    "rho_decile_debiased_accuracy": rho_debiased,
                    "rho_decile_valid_accuracy_non_tie": rho_valid_non_tie,
                    "rho_decile_effective_accuracy_non_tie": rho_effective_non_tie,
                    "rho_decile_debiased_accuracy_non_tie": rho_debiased_non_tie,
                    "top_minus_bottom_valid_accuracy": valid_diff,
                    "valid_difference_ci_low": valid_low,
                    "valid_difference_ci_high": valid_high,
                    "valid_bootstrap_reps_used": valid_reps,
                    "top_minus_bottom_effective_accuracy": eff_diff,
                    "effective_difference_ci_low": eff_low,
                    "effective_difference_ci_high": eff_high,
                    "effective_bootstrap_reps_used": eff_reps,
                    "top_minus_bottom_debiased_accuracy": deb_diff,
                    "debiased_difference_ci_low": deb_low,
                    "debiased_difference_ci_high": deb_high,
                    "debiased_bootstrap_reps_used": deb_reps,
                    "top_minus_bottom_effective_accuracy_non_tie": eff_nt_diff,
                    "effective_non_tie_difference_ci_low": eff_nt_low,
                    "effective_non_tie_difference_ci_high": eff_nt_high,
                    "effective_non_tie_bootstrap_reps_used": eff_nt_reps,
                    "top_minus_bottom_debiased_accuracy_non_tie": deb_nt_diff,
                    "debiased_non_tie_difference_ci_low": deb_nt_low,
                    "debiased_non_tie_difference_ci_high": deb_nt_high,
                    "debiased_non_tie_bootstrap_reps_used": deb_nt_reps,
                    "calibration_target": "debiased_correct; c=0 is incorrect",
                    "calibration_train_pairs": len(train),
                    "calibration_test_pairs": len(test),
                    "calibration_cross_pairs_excluded": len(cross),
                    "calibration_train_unique_snippets": len(train_snippets),
                    "calibration_test_unique_snippets": len(test_snippets),
                    "calibration_snippet_overlap": len(train_snippets & test_snippets),
                    "isotonic_ece_10_equal_frequency_bins": ece,
                    "isotonic_auroc": auroc,
                    "raw_abs_c_auroc": raw_auroc,
                }
            )
            if scope == "overall":
                for _, row in dec.iterrows():
                    figure_rows.append(row.to_dict())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(target / "B_calibration_summary.csv", index=False)
    figure_data = pd.DataFrame(figure_rows)
    figure_data.to_csv(target / "B_calibration_figure_data.csv", index=False)

    fig, axes = plt.subplots(1, 5, figsize=(13.2, 2.9), sharex=True, sharey=True)
    for axis, (model_key, frame) in zip(
        axes, figure_data.groupby("model_key", sort=True)
    ):
        frame = frame.sort_values("decile")
        axis.plot(frame["decile"], frame["valid_accuracy"], marker="o", label="Valid")
        axis.plot(
            frame["decile"],
            frame["effective_accuracy"],
            marker="s",
            label="Effective",
        )
        axis.plot(
            frame["decile"],
            frame["debiased_accuracy"],
            marker="^",
            label="Debiased",
        )
        axis.set_title(MODEL_LABELS.get(model_key, model_key), fontsize=9)
        axis.set_xticks([1, 5, 10])
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Accuracy / preference")
    for axis in axes:
        axis.set_xlabel("|c| decile")
        axis.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(target / "B_content_calibration.pdf", bbox_inches="tight")
    fig.savefig(target / "B_content_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"decile_rows": len(deciles), "summary_rows": len(summary), "summary": summary}


def analysis_c(battery: pd.DataFrame, grid: pd.DataFrame) -> dict:
    target = OUT / "C_bc_joint"
    target.mkdir(parents=True, exist_ok=True)
    joint = battery[
        [
            "model_key",
            "model",
            "pair_id",
            "language",
            "difficulty",
            "snippet_i",
            "snippet_j",
            "abs_b",
            "abs_c",
            "valid",
            "correct",
            "content_tie",
        ]
    ].copy()
    joint["boundary_abs_c_eq_abs_b"] = joint["abs_c"].eq(joint["abs_b"])
    joint["identity_predicted_valid"] = joint["abs_c"].gt(joint["abs_b"])
    joint["identity_violation_nonboundary"] = (
        ~joint["boundary_abs_c_eq_abs_b"]
        & joint["identity_predicted_valid"].ne(joint["valid"])
    )
    joint["identity_mismatch_conservative_boundary_included"] = (
        joint["identity_predicted_valid"].ne(joint["valid"])
    )
    joint.to_csv(target / "C_bc_joint.csv", index=False)

    summaries = []
    scope_groups = [
        ("overall", "all", frame)
        for _, frame in joint.groupby("model_key", sort=True)
    ]
    scope_groups.extend(
        ("language", language, frame)
        for (_, language), frame in joint.groupby(["model_key", "language"], sort=True)
    )
    for scope, scope_value, group in scope_groups:
        summaries.append(
            {
                "run": "rq1_model_battery_3lang_20260723",
                "model_key": group.iloc[0]["model_key"],
                "model": group.iloc[0]["model"],
                "scope": scope,
                "language": scope_value,
                "n_pairs": len(group),
                "median_abs_b": group["abs_b"].median(),
                "median_abs_c": group["abs_c"].median(),
                "abs_c_gt_abs_b_pairs": int(group["identity_predicted_valid"].sum()),
                "abs_c_gt_abs_b_rate": group["identity_predicted_valid"].mean(),
                "strict_valid_pairs": int(group["valid"].sum()),
                "strict_valid_rate": group["valid"].mean(),
                "boundary_pairs": int(group["boundary_abs_c_eq_abs_b"].sum()),
                "nonboundary_identity_violations": int(
                    group["identity_violation_nonboundary"].sum()
                ),
                "conservative_boundary_inclusive_mismatches": int(
                    group["identity_mismatch_conservative_boundary_included"].sum()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(target / "C_bc_summary.csv", index=False)

    grid_only = grid[grid["experiment"].eq("grid")].copy()
    trajectory = (
        grid_only.groupby(["model", "language", "condition"], sort=True)
        .agg(
            n_pairs=("pair_id", "size"),
            median_abs_b=("abs_b", "median"),
            median_abs_c=("abs_c", "median"),
            strict_valid_rate=("strict_valid", "mean"),
            boundary_rate=("logit_boundary", "mean"),
        )
        .reset_index()
    )
    trajectory.to_csv(target / "C_grid_condition_trajectory.csv", index=False)

    fig, axes = plt.subplots(1, 5, figsize=(13.2, 2.8))
    for axis, (model_key, frame) in zip(axes, joint.groupby("model_key", sort=True)):
        max_value = float(np.quantile(np.r_[frame["abs_b"], frame["abs_c"]], 0.995))
        axis.hexbin(
            frame["abs_c"],
            frame["abs_b"],
            gridsize=38,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        axis.plot([0, max_value], [0, max_value], color="white", linewidth=1)
        axis.set_xlim(0, max_value)
        axis.set_ylim(0, max_value)
        axis.set_title(MODEL_LABELS.get(model_key, model_key), fontsize=9)
        axis.set_xlabel("|c|")
    axes[0].set_ylabel("|b|")
    fig.tight_layout()
    fig.savefig(target / "C_bc_joint_density.pdf", bbox_inches="tight")
    fig.savefig(target / "C_bc_joint_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "joint_rows": len(joint),
        "nonboundary_violations": int(joint["identity_violation_nonboundary"].sum()),
        "boundary_pairs": int(joint["boundary_abs_c_eq_abs_b"].sum()),
        "boundary_inclusive_mismatches": int(
            joint["identity_mismatch_conservative_boundary_included"].sum()
        ),
    }


def analysis_d(battery: pd.DataFrame) -> dict:
    target = OUT / "D_first_position"
    target.mkdir(parents=True, exist_ok=True)
    battery = battery.copy()
    battery["first_position_choices_per_pair"] = (
        battery["ab_first_choice"].astype(int) + battery["ba_first_choice"].astype(int)
    )
    rows = []
    group_specs = [
        ("language", ["model_key", "language"]),
        ("language_difficulty", ["model_key", "language", "difficulty"]),
    ]
    for scope, columns in group_specs:
        for group_index, (keys, group) in enumerate(battery.groupby(columns, sort=True)):
            if not isinstance(keys, tuple):
                keys = (keys,)
            first_choices = int(group["first_position_choices_per_pair"].sum())
            n_calls = 2 * len(group)
            pair_values = group["first_position_choices_per_pair"].to_numpy(dtype=float) / 2
            low, high = bootstrap_mean_ci(
                pair_values, BOOTSTRAP_REPS, SEED + 5000 + group_index
            )
            row = {
                "run": "rq1_model_battery_3lang_20260723",
                "scope": scope,
                "model_key": keys[0],
                "model": group.iloc[0]["model"],
                "language": keys[1],
                "difficulty": keys[2] if len(keys) > 2 else "all",
                "n_pairs": len(group),
                "n_calls": n_calls,
                "first_position_choices": first_choices,
                "first_position_rate": first_choices / n_calls,
                "pair_cluster_bootstrap_ci_low": low,
                "pair_cluster_bootstrap_ci_high": high,
                "bootstrap_reps": BOOTSTRAP_REPS,
            }
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(target / "D_first_position_by_language.csv", index=False)
    return {"rows": len(result), "language_rows": int(result["scope"].eq("language").sum())}


def prepare_battery() -> pd.DataFrame:
    pairs = pd.read_csv(BATTERY_PAIR_PATH)
    metadata = pd.read_csv(BATTERY_META_PATH)[
        ["protocol_pair_id", "snippet_i", "snippet_j", "language", "difficulty"]
    ].rename(columns={"protocol_pair_id": "pair_id"})
    merged = pairs.merge(
        metadata,
        on=["pair_id", "language", "difficulty"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != 45000 or merged[["snippet_i", "snippet_j"]].isna().any().any():
        raise ValueError("Battery pair metadata merge failed")
    merged["snippet_i_key"] = merged["language"] + "::" + merged["snippet_i"]
    merged["snippet_j_key"] = merged["language"] + "::" + merged["snippet_j"]
    merged["abs_c"] = merged["content_margin"].abs()
    merged["abs_b"] = merged["position_margin"].abs()
    return merged


def main() -> None:
    require_ga0()
    grid = pd.read_csv(GRID_PAIR_PATH)
    battery = prepare_battery()
    results = {
        "new_model_calls": 0,
        "runs_kept_separate": True,
        "A": analysis_a(grid),
        "B": analysis_b(battery),
        "C": analysis_c(battery, grid),
        "D": analysis_d(battery),
    }
    serializable = {
        "new_model_calls": 0,
        "runs_kept_separate": True,
        "A": {key: value for key, value in results["A"].items() if not isinstance(value, pd.DataFrame)},
        "B": {key: value for key, value in results["B"].items() if not isinstance(value, pd.DataFrame)},
        "C": results["C"],
        "D": results["D"],
    }
    (OUT / "audit/ANALYSIS_EXECUTION.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()

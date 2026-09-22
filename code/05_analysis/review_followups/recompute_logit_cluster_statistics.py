#!/usr/bin/env python3
"""Recompute the Section 6.3 valid/swap margin tests with clustered uncertainty."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path("/ANON/experiment_root")
SOURCE = ROOT / "results/grounded_protocol_3lang_20260721/analysis/logit_decomposition/LOGIT_PAIR_LEVEL.csv"
DEFAULT_OUT = ROOT / "results/fse2027_final_verification/logit_cluster_statistics.csv"
ORIGINAL_SOURCE = ROOT / "results/protocol_unified_3lang_20260720/a1/analysis/logit_component_mannwhitney.csv"
ORIGINAL_OUT = ROOT / "results/fse2027_final_verification/original_logit_mannwhitney_audit.csv"


def holm(values: pd.Series) -> pd.Series:
    p = values.to_numpy(float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return pd.Series(adjusted, index=values.index)


def aggregate_pairs(part: pd.DataFrame, value: str) -> pd.DataFrame:
    work = part.assign(status=np.where(part["strict_valid"], "valid", "swap"))
    agg = (
        work.groupby(["pair_id", "snippet_i", "snippet_j", "status"], as_index=False)[value]
        .agg(["sum", "count"])
        .reset_index()
    )
    rows = []
    for key, group in agg.groupby(["pair_id", "snippet_i", "snippet_j"], sort=False):
        row = {"pair_id": key[0], "snippet_i": key[1], "snippet_j": key[2]}
        for status in ("swap", "valid"):
            hit = group[group["status"].eq(status)]
            row[f"{status}_sum"] = float(hit["sum"].sum())
            row[f"{status}_n"] = int(hit["count"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_effect(agg: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    ss = weights @ agg["swap_sum"].to_numpy(float)
    sn = weights @ agg["swap_n"].to_numpy(float)
    vs = weights @ agg["valid_sum"].to_numpy(float)
    vn = weights @ agg["valid_n"].to_numpy(float)
    return ss / sn - vs / vn


def pair_cluster_bootstrap(agg: pd.DataFrame, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(agg)
    out = []
    for start in range(0, reps, 250):
        size = min(250, reps - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=size)
        out.append(weighted_effect(agg, weights))
    return np.concatenate(out)


def snippet_pigeonhole_bootstrap(agg: pd.DataFrame, reps: int, seed: int) -> np.ndarray:
    """Crossed-cluster Bayesian bootstrap using independent snippet weights.

    Each observation receives the product of the two endpoint snippet weights.
    This preserves every rendering observation attached to a sampled pair while
    reflecting reuse of either endpoint across pairs.
    """
    snippets = pd.Index(sorted(set(agg["snippet_i"]) | set(agg["snippet_j"])))
    left = snippets.get_indexer(agg["snippet_i"])
    right = snippets.get_indexer(agg["snippet_j"])
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, reps, 250):
        size = min(250, reps - start)
        node_weights = rng.exponential(1.0, size=(size, len(snippets)))
        pair_weights = node_weights[:, left] * node_weights[:, right]
        out.append(weighted_effect(agg, pair_weights))
    return np.concatenate(out)


def summarize_bootstrap(draws: np.ndarray, estimate: float, reps: int) -> dict[str, float | int]:
    centered = draws - estimate
    extreme = int(np.count_nonzero(np.abs(centered) >= abs(estimate)))
    return {
        "effect_ci_low": float(np.quantile(draws, 0.025)),
        "effect_ci_high": float(np.quantile(draws, 0.975)),
        "raw_p": float((extreme + 1) / (reps + 1)),
        "resamples": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--original-output", type=Path, default=ORIGINAL_OUT)
    args = parser.parse_args()

    data = pd.read_csv(args.source)
    nonboundary = data.loc[~data["logit_boundary"].astype(bool)].copy()
    if len(data) != 108_000 or len(nonboundary) != 99_832:
        raise RuntimeError(f"Unexpected row counts: all={len(data)}, nonboundary={len(nonboundary)}")

    rows: list[dict] = []
    grouped = nonboundary.groupby(["experiment", "model"], sort=True)
    for group_idx, ((experiment, model), part) in enumerate(grouped):
        for metric_idx, value in enumerate(("abs_b", "abs_c")):
            swap = part.loc[~part["strict_valid"].astype(bool), value].to_numpy(float)
            valid = part.loc[part["strict_valid"].astype(bool), value].to_numpy(float)
            u = mannwhitneyu(swap, valid, alternative="two-sided", method="asymptotic")
            cliffs = 2 * float(u.statistic) / (len(swap) * len(valid)) - 1
            estimate = float(swap.mean() - valid.mean())
            common = {
                "group": f"{experiment}::{model}",
                "experiment": experiment,
                "model": model,
                "metric": value,
                "n_observations": len(part),
                "n_swap": len(swap),
                "n_valid": len(valid),
                "swap_mean": float(swap.mean()),
                "swap_median": float(np.median(swap)),
                "valid_mean": float(valid.mean()),
                "valid_median": float(np.median(valid)),
                "effect_definition": "swap_mean_minus_valid_mean",
                "effect_size_mean_difference": estimate,
                "cliffs_delta_swap_vs_valid": cliffs,
                "boundary_exclusion": "logit_boundary == False; boundary is abs_b == abs_c",
            }
            rows.append({
                **common,
                "method": "naive_mann_whitney",
                "effect_ci_low": np.nan,
                "effect_ci_high": np.nan,
                "raw_p": float(u.pvalue),
                "resamples": 0,
                "seed": np.nan,
                "cluster_unit": "individual rendering/perturbation observation",
                "cluster_count": len(part),
            })

            agg = aggregate_pairs(part, value)
            pair_draws = pair_cluster_bootstrap(
                agg, args.resamples, args.seed + group_idx * 100 + metric_idx
            )
            rows.append({
                **common,
                "method": "pair_cluster_bootstrap",
                **summarize_bootstrap(pair_draws, estimate, args.resamples),
                "seed": args.seed + group_idx * 100 + metric_idx,
                "cluster_unit": "pair_id; all conditions retained within pair",
                "cluster_count": int(agg["pair_id"].nunique()),
            })

            snippet_draws = snippet_pigeonhole_bootstrap(
                agg, args.resamples, args.seed + 10_000 + group_idx * 100 + metric_idx
            )
            rows.append({
                **common,
                "method": "crossed_snippet_bayesian_bootstrap",
                **summarize_bootstrap(snippet_draws, estimate, args.resamples),
                "seed": args.seed + 10_000 + group_idx * 100 + metric_idx,
                "cluster_unit": "crossed snippet_i/snippet_j; product exponential weights",
                "cluster_count": int(len(set(agg["snippet_i"]) | set(agg["snippet_j"]))),
            })

    result = pd.DataFrame(rows)
    result["holm_p"] = np.nan
    for method, idx in result.groupby("method").groups.items():
        result.loc[idx, "holm_p"] = holm(result.loc[idx, "raw_p"])
    result["direction_expected"] = np.where(result["metric"].eq("abs_b"), "positive", "negative")
    result["direction_maintained"] = np.where(
        result["metric"].eq("abs_b"),
        result["effect_size_mean_difference"].gt(0),
        result["effect_size_mean_difference"].lt(0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    original = pd.read_csv(ORIGINAL_SOURCE)
    original = original[
        original["language"].eq("all")
        & original["condition"].eq("monokai_dark__fs20__wrap80__lnon")
    ].copy()
    if len(original) != 4:
        raise RuntimeError(f"Expected four historical baseline tests, found {len(original)}")
    original["holm_p_over_four"] = holm(original["p_value"])
    original["observation_unit"] = "pair; one baseline condition per model"
    original["boundary_excluded"] = False
    original["source_file"] = str(ORIGINAL_SOURCE)
    original.to_csv(args.original_output, index=False)


if __name__ == "__main__":
    main()

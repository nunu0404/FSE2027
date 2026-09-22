#!/usr/bin/env python3
"""Compute pairwise accuracy for classical score-based baselines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(group: pd.DataFrame) -> dict[str, float | int]:
    all_pairs = len(group)
    valid = group[group["is_valid_score_pair"]].copy()
    correct = int(valid["is_correct"].sum())
    out: dict[str, float | int] = {
        "all_pairs": all_pairs,
        "valid_pairs": int(len(valid)),
        "correct_pairs": correct,
        "pairwise_accuracy_valid": float(correct / len(valid)) if len(valid) else np.nan,
        "effective_pairwise_accuracy": float(correct / all_pairs) if all_pairs else np.nan,
        "score_tie_rate": float((~group["is_valid_score_pair"]).sum() / all_pairs) if all_pairs else np.nan,
        "strict_swap_error_rate": np.nan,
        "parse_failure_rate": np.nan,
    }
    for level in ["easy", "medium", "hard"]:
        sub = valid[valid["difficulty"] == level]
        out[f"{level}_accuracy_valid"] = float(sub["is_correct"].mean()) if len(sub) else np.nan
        out[f"{level}_n"] = int(len(sub))
    return out


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="experiments/rq0_viability/data/pairs/fixed_pair_sets.csv")
    parser.add_argument("--predictions", default="experiments/rq0_viability/outputs/classical_baselines/classical_regression_predictions.csv")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    root = repo_root / "experiments/rq0_viability"
    pairs = pd.read_csv(repo_root / args.pairs)
    preds = pd.read_csv(repo_root / args.predictions)
    preds = preds[preds["target"] == "z_normalized"].copy()

    pair_level_frames = []
    metric_rows = []
    for (split_id, model), pred_group in preds.groupby(["split_id", "model"], sort=True):
        split_pairs = pairs[pairs["split_id"] == split_id].copy()
        if split_pairs.empty:
            continue
        score_map = pred_group.set_index("rq0_id")["predicted_score"]
        split_pairs["predicted_score_i"] = split_pairs["snippet_i"].map(score_map)
        split_pairs["predicted_score_j"] = split_pairs["snippet_j"].map(score_map)
        split_pairs = split_pairs.dropna(subset=["predicted_score_i", "predicted_score_j"]).copy()
        split_pairs["model_preference"] = np.where(
            split_pairs["predicted_score_i"] > split_pairs["predicted_score_j"],
            split_pairs["snippet_i"],
            np.where(split_pairs["predicted_score_j"] > split_pairs["predicted_score_i"], split_pairs["snippet_j"], ""),
        )
        split_pairs["is_valid_score_pair"] = split_pairs["model_preference"] != ""
        split_pairs["is_correct"] = split_pairs["model_preference"] == split_pairs["human_preference"]
        split_pairs["model_group"] = "classical_feature"
        split_pairs["model"] = model
        split_pairs["input"] = "features"
        split_pairs["training_setting"] = "supervised"
        pair_level_frames.append(split_pairs)
        metric = summarize(split_pairs)
        first = split_pairs.iloc[0]
        metric_rows.append(
            {
                "split_id": split_id,
                "split_type": first["split_type"],
                "seed": first["seed"],
                "fold": first["fold"],
                "model_group": "classical_feature",
                "model": model,
                "input": "features",
                "training_setting": "supervised",
                **metric,
            }
        )

    pair_level = pd.concat(pair_level_frames, ignore_index=True) if pair_level_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    out_pair_dir = root / "outputs/pair_level_results"
    out_metric_dir = root / "outputs/metrics"
    out_pair_dir.mkdir(parents=True, exist_ok=True)
    out_metric_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_pair_dir / "classical_pairwise_results.csv"
    metrics_path = out_metric_dir / "classical_pairwise_metrics.csv"
    pair_level.to_csv(pair_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    summary = (
        metrics.groupby(["split_type", "model"], dropna=False)[
            ["pairwise_accuracy_valid", "effective_pairwise_accuracy", "easy_accuracy_valid", "medium_accuracy_valid", "hard_accuracy_valid"]
        ]
        .agg(["mean", "std", "count"])
    )
    summary_path = root / "outputs/tables/classical_pairwise_summary.csv"
    summary.to_csv(summary_path)
    write_json(
        out_metric_dir / "classical_pairwise_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pairs": str((repo_root / args.pairs).resolve()),
            "predictions": str((repo_root / args.predictions).resolve()),
            "outputs": {"pair_level": str(pair_path), "metrics": str(metrics_path), "summary": str(summary_path)},
            "pair_level_rows": int(len(pair_level)),
            "metric_rows": int(len(metrics)),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

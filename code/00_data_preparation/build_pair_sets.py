#!/usr/bin/env python3
"""Build fixed test-only pair sets for RQ0 pairwise evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rq0_build_pairs")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def difficulty(abs_diff: float) -> str:
    if abs_diff >= 1.0:
        return "easy"
    if abs_diff >= 0.5:
        return "medium"
    if abs_diff >= 0.2:
        return "hard"
    return "exclude"


def sample_rows(rows: list[dict[str, object]], target_per_difficulty: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    selected = []
    for level in ["easy", "medium", "hard"]:
        level_rows = [row for row in rows if row["difficulty"] == level]
        if len(level_rows) > target_per_difficulty:
            level_rows = rng.sample(level_rows, target_per_difficulty)
        selected.extend(level_rows)
    selected.sort(key=lambda row: (str(row["difficulty"]), str(row["pair_id"])))
    return selected


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/rq0_viability/configs/rq0_main.yaml")
    parser.add_argument("--dataset", default="experiments/rq0_viability/data/processed/pooled_313_processed.csv")
    parser.add_argument("--splits", default="experiments/rq0_viability/data/splits/split_assignments.csv")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = yaml.safe_load((repo_root / args.config).read_text(encoding="utf-8"))
    root = repo_root / config["experiment"]["root"]
    logger = setup_logger(root / "outputs/logs/build_pair_sets.log")
    dataset = pd.read_csv(repo_root / args.dataset)
    assignments = pd.read_csv(repo_root / args.splits)
    by_id = dataset.set_index("rq0_id")
    target = int(config["pairs"]["target_per_difficulty"])
    compact_target = int(config["pairs"]["compact_per_difficulty"])

    all_rows: list[dict[str, object]] = []
    compact_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for split_id in sorted(assignments["split_id"].unique()):
        split = assignments[assignments["split_id"] == split_id]
        meta = split.iloc[0]
        seed = int(meta["seed"])
        sample_seed = seed if seed >= 0 else 42
        test_ids = sorted(split[split["role"] == "test"]["rq0_id"].tolist())
        candidate_rows: list[dict[str, object]] = []
        for a, b in itertools.combinations(test_ids, 2):
            row_a = by_id.loc[a]
            row_b = by_id.loc[b]
            score_a = float(row_a["human_score_z_within_dataset"])
            score_b = float(row_b["human_score_z_within_dataset"])
            diff = score_a - score_b
            abs_diff = abs(diff)
            level = difficulty(abs_diff)
            if level == "exclude" or diff == 0:
                continue
            preferred = a if diff > 0 else b
            candidate_rows.append(
                {
                    "pair_id": f"{split_id}__{a}__{b}",
                    "split_id": split_id,
                    "split_type": meta["split_type"],
                    "seed": seed,
                    "fold": int(meta["fold"]),
                    "snippet_i": a,
                    "snippet_j": b,
                    "dataset_name_i": row_a["dataset_name"],
                    "dataset_name_j": row_b["dataset_name"],
                    "difficulty": level,
                    "human_score_i_z": score_a,
                    "human_score_j_z": score_b,
                    "abs_z_diff": abs_diff,
                    "human_preference": preferred,
                    "preference_score_basis": "human_score_z_within_dataset",
                }
            )
        selected = sample_rows(candidate_rows, target, sample_seed)
        compact_selected = sample_rows(candidate_rows, compact_target, sample_seed)
        all_rows.extend(selected)
        if split_id == "pooled_z_seed42_fold0":
            compact_rows.extend(compact_selected)
        counts_before = pd.Series([r["difficulty"] for r in candidate_rows]).value_counts().to_dict()
        counts_after = pd.Series([r["difficulty"] for r in selected]).value_counts().to_dict()
        summary_rows.append(
            {
                "split_id": split_id,
                "split_type": meta["split_type"],
                "seed": seed,
                "fold": int(meta["fold"]),
                "test_snippets": len(test_ids),
                "candidate_pairs": len(candidate_rows),
                "selected_pairs": len(selected),
                "candidate_easy": counts_before.get("easy", 0),
                "candidate_medium": counts_before.get("medium", 0),
                "candidate_hard": counts_before.get("hard", 0),
                "selected_easy": counts_after.get("easy", 0),
                "selected_medium": counts_after.get("medium", 0),
                "selected_hard": counts_after.get("hard", 0),
            }
        )
        logger.info("split=%s candidates=%d selected=%d counts=%s", split_id, len(candidate_rows), len(selected), counts_after)

    pair_dir = root / "data/pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pair_dir / "fixed_pair_sets.csv"
    compact_path = pair_dir / "compact_pooled_seed42_fold0_pairs.csv"
    summary_path = root / "outputs/tables/pair_set_summary.csv"
    pd.DataFrame(all_rows).to_csv(pairs_path, index=False)
    pd.DataFrame(compact_rows).to_csv(compact_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preference_rule": "All pairwise preferences use human_score_z_within_dataset; raw scores are never used for cross-dataset comparison.",
        "pair_scope_rule": "Pairs are generated only among test snippets within the same split.",
        "target_per_difficulty": target,
        "compact_per_difficulty": compact_target,
        "outputs": {
            "fixed_pair_sets": str(pairs_path),
            "compact_pair_set": str(compact_path),
            "summary": str(summary_path),
        },
        "total_selected_pairs": len(all_rows),
        "compact_selected_pairs": len(compact_rows),
    }
    write_json(pair_dir / "pair_manifest.json", manifest)
    logger.info("wrote pairs=%s rows=%d", pairs_path, len(all_rows))
    logger.info("wrote compact=%s rows=%d", compact_path, len(compact_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

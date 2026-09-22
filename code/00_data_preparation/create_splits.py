#!/usr/bin/env python3
"""Create leakage-aware RQ0 train/test split manifests."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rq0_create_splits")
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


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def group_key(row: pd.Series) -> str:
    raw_group = row.get("duplicate_group_id", "")
    group = "" if pd.isna(raw_group) else str(raw_group)
    return group if group else str(row["rq0_id"])


def add_assignment(
    rows: list[dict[str, object]],
    split_id: str,
    split_type: str,
    seed: int,
    fold: int,
    role: str,
    frame: pd.DataFrame,
    train_datasets: str = "",
    test_dataset: str = "",
) -> None:
    for _, row in frame.iterrows():
        rows.append(
            {
                "split_id": split_id,
                "split_type": split_type,
                "seed": seed,
                "fold": fold,
                "role": role,
                "rq0_id": row["rq0_id"],
                "dataset_name": row["dataset_name"],
                "tertiary_label": row["tertiary_label"],
                "duplicate_group_key": row["split_group_key"],
                "train_datasets": train_datasets,
                "test_dataset": test_dataset,
            }
        )


def make_stratified_group_folds(
    df: pd.DataFrame,
    split_type: str,
    seed: int,
    n_splits: int,
    stratify_col: str,
    split_prefix: str,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df[stratify_col].astype(str)
    groups = df["split_group_key"].astype(str)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups)):
        split_id = f"{split_prefix}_seed{seed}_fold{fold}"
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        overlap = set(train["split_group_key"]) & set(test["split_group_key"])
        if overlap:
            raise RuntimeError(f"Duplicate group leakage in {split_id}: {sorted(overlap)[:5]}")
        logger.info(
            "split=%s train=%d test=%d stratify_counts_test=%s",
            split_id,
            len(train),
            len(test),
            test[stratify_col].value_counts().sort_index().to_dict(),
        )
        add_assignment(rows, split_id, split_type, seed, fold, "train", train)
        add_assignment(rows, split_id, split_type, seed, fold, "test", test)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/rq0_viability/configs/rq0_main.yaml")
    parser.add_argument("--dataset", default="experiments/rq0_viability/data/processed/pooled_313_processed.csv")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = yaml.safe_load((repo_root / args.config).read_text(encoding="utf-8"))
    root = repo_root / config["experiment"]["root"]
    logger = setup_logger(root / "outputs/logs/create_splits.log")
    df = pd.read_csv(repo_root / args.dataset)
    df["split_group_key"] = df.apply(group_key, axis=1)
    df["pooled_stratify_label"] = df["dataset_name"].astype(str) + "__" + df["tertiary_label"].astype(str)

    seeds = list(config["experiment"]["seeds"])
    within_folds = int(config["splits"]["within_dataset_folds"])
    pooled_folds = int(config["splits"]["pooled_folds"])
    all_rows: list[dict[str, object]] = []

    for seed in seeds:
        for dataset_name, group in df.groupby("dataset_name", sort=True):
            all_rows.extend(
                make_stratified_group_folds(
                    group.reset_index(drop=True),
                    split_type="within_dataset_5fold",
                    seed=seed,
                    n_splits=within_folds,
                    stratify_col="tertiary_label",
                    split_prefix=f"within_{dataset_name.lower()}",
                    logger=logger,
                )
            )
        all_rows.extend(
            make_stratified_group_folds(
                df.reset_index(drop=True),
                split_type="pooled_stratified_5fold",
                seed=seed,
                n_splits=pooled_folds,
                stratify_col="pooled_stratify_label",
                split_prefix="pooled_z",
                logger=logger,
            )
        )

    for held_out in sorted(df["dataset_name"].unique()):
        train = df[df["dataset_name"] != held_out].copy()
        test = df[df["dataset_name"] == held_out].copy()
        split_id = f"lodo_test_{held_out.lower()}"
        train_datasets = "+".join(sorted(train["dataset_name"].unique()))
        logger.info("split=%s train=%d test=%d", split_id, len(train), len(test))
        add_assignment(
            all_rows,
            split_id,
            "leave_one_dataset_out",
            seed=-1,
            fold=0,
            role="train",
            frame=train,
            train_datasets=train_datasets,
            test_dataset=held_out,
        )
        add_assignment(
            all_rows,
            split_id,
            "leave_one_dataset_out",
            seed=-1,
            fold=0,
            role="test",
            frame=test,
            train_datasets=train_datasets,
            test_dataset=held_out,
        )

    out = pd.DataFrame(all_rows)
    split_dir = root / "data/splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = split_dir / "split_assignments.csv"
    out.to_csv(assignments_path, index=False)

    summary = (
        out.groupby(["split_type", "split_id", "role"])
        .size()
        .reset_index(name="n")
        .pivot_table(index=["split_type", "split_id"], columns="role", values="n", fill_value=0)
        .reset_index()
    )
    summary_path = root / "outputs/tables/split_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    leakage_checks = []
    for split_id, group in out.groupby("split_id"):
        train_groups = set(group[group["role"] == "train"]["duplicate_group_key"])
        test_groups = set(group[group["role"] == "test"]["duplicate_group_key"])
        leaked = sorted(g for g in train_groups & test_groups if g)
        leakage_checks.append({"split_id": split_id, "leaked_group_count": len(leaked), "leaked_groups": leaked})
    leak_path = root / "outputs/tables/split_leakage_check.json"
    write_json(leak_path, leakage_checks)
    if any(item["leaked_group_count"] for item in leakage_checks):
        raise RuntimeError(f"Split leakage detected. See {leak_path}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str((repo_root / args.dataset).resolve()),
        "seeds": seeds,
        "outputs": {
            "assignments": str(assignments_path),
            "summary": str(summary_path),
            "leakage_check": str(leak_path),
        },
        "split_counts": out["split_id"].nunique(),
    }
    write_json(split_dir / "split_manifest.json", manifest)
    logger.info("wrote assignments=%s rows=%d split_ids=%d", assignments_path, len(out), out["split_id"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

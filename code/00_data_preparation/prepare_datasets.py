#!/usr/bin/env python3
"""Prepare unified RQ0 readability datasets without modifying raw sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rq0_prepare_datasets")
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def normalize_exact_code(value: Any) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def normalize_formatting_only(value: Any) -> str:
    text = normalize_exact_code(value)
    return re.sub(r"\s+", "", text)


def group_ids(values: pd.Series, prefix: str) -> pd.Series:
    counts = values.value_counts(dropna=False)
    duplicate_keys = {k for k, v in counts.items() if v > 1}
    mapping: dict[Any, str] = {}
    next_id = 1
    output = []
    for value in values:
        if value in duplicate_keys:
            if value not in mapping:
                mapping[value] = f"{prefix}{next_id:04d}"
                next_id += 1
            output.append(mapping[value])
        else:
            output.append("")
    return pd.Series(output, index=values.index)


def labels_for_dataset(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    scores = out["human_mean_score"].astype(float)
    mean = scores.mean()
    std = scores.std(ddof=0)
    out["human_score_z_within_dataset"] = 0.0 if std == 0 else (scores - mean) / std
    min_score = scores.min()
    max_score = scores.max()
    denom = max_score - min_score
    out["human_score_minmax_within_dataset"] = 0.0 if denom == 0 else (scores - min_score) / denom

    median = scores.median()
    out["binary_label_median"] = np.where(scores >= median, "readable", "unreadable")

    q33 = scores.quantile(1 / 3)
    q67 = scores.quantile(2 / 3)
    strict = np.full(len(out), "discard", dtype=object)
    strict[scores <= q33] = "unreadable"
    strict[scores >= q67] = "readable"
    out["binary_label_tertile_strict"] = strict

    tertiary = np.full(len(out), "mid", dtype=object)
    tertiary[scores <= q33] = "low"
    tertiary[scores >= q67] = "high"
    out["tertiary_label"] = tertiary
    out["dataset_score_mean"] = mean
    out["dataset_score_std"] = std
    out["dataset_score_q33"] = q33
    out["dataset_score_q67"] = q67
    return out


def load_source(source_key: str, spec: dict[str, Any], columns: dict[str, str], repo_root: Path) -> pd.DataFrame:
    path = repo_root / spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw source for {source_key}: {path}")
    df = pd.read_csv(path)
    required = [columns["id"], columns["code"], columns["score"]]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column: {col}")
    dataset_name = spec["dataset_name"]
    out = pd.DataFrame()
    out["source_key"] = source_key
    out["source_path"] = str(path)
    out["source_row_index"] = np.arange(len(df))
    out["snippet_id"] = df[columns["id"]]
    out["dataset_name"] = dataset_name
    out["raw_code"] = df[columns["code"]].map(normalize_exact_code)
    out["human_mean_score"] = df[columns["score"]].astype(float)
    out["language"] = df.get(columns.get("language", "language"), "java")
    out["LOC"] = df.get(columns.get("loc", "LOC"), np.nan)
    out["cyclomatic_complexity"] = df.get(columns.get("cyclomatic_complexity", "cyclomatic_complexity"), np.nan)
    out["min_score"] = df.get("min_score", np.nan)
    out["max_score"] = df.get("max_score", np.nan)
    out["code_type"] = df.get("code_type", "")
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = (repo_root / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    processed_dir = repo_root / config["processed_dir"]
    tables_dir = repo_root / config["tables_dir"]
    logs_dir = repo_root / config["logs_dir"]
    logger = setup_logger(logs_dir / "prepare_datasets.log")
    logger.info("repo_root=%s", repo_root)
    logger.info("config=%s", config_path)

    frames = []
    for source_key, spec in config["raw_sources"].items():
        frame = load_source(source_key, spec, config["columns"], repo_root)
        logger.info("loaded source=%s rows=%d path=%s", source_key, len(frame), spec["path"])
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "rq0_id", [f"rq0_{i:04d}" for i in range(len(df))])

    df["exact_code_hash"] = df["raw_code"].map(sha256_text)
    df["formatting_normalized_hash"] = df["raw_code"].map(normalize_formatting_only).map(sha256_text)
    df["exact_duplicate_group_id"] = group_ids(df["exact_code_hash"], "exact_dup_")
    df["formatting_duplicate_group_id"] = group_ids(df["formatting_normalized_hash"], "fmt_dup_")
    df["duplicate_group_id"] = df["formatting_duplicate_group_id"].where(
        df["formatting_duplicate_group_id"] != "",
        df["exact_duplicate_group_id"],
    )

    df = (
        df.groupby("dataset_name", group_keys=False, sort=False)
        .apply(labels_for_dataset, include_groups=False)
        .reset_index(drop=True)
    )
    # groupby/apply drops the grouping column with include_groups=False.
    # Restore it from the original row order.
    df["dataset_name"] = pd.concat(frames, ignore_index=True)["dataset_name"].values
    df["source_metadata"] = df.apply(
        lambda row: json.dumps(
            {
                "source_key": row["source_key"],
                "source_path": row["source_path"],
                "source_row_index": int(row["source_row_index"]),
                "original_snippet_id": str(row["snippet_id"]),
            },
            ensure_ascii=True,
        ),
        axis=1,
    )
    df["num_annotators"] = np.nan

    ordered_columns = [
        "rq0_id",
        "snippet_id",
        "dataset_name",
        "raw_code",
        "human_mean_score",
        "human_score_z_within_dataset",
        "human_score_minmax_within_dataset",
        "binary_label_median",
        "binary_label_tertile_strict",
        "tertiary_label",
        "num_annotators",
        "language",
        "LOC",
        "cyclomatic_complexity",
        "min_score",
        "max_score",
        "code_type",
        "exact_code_hash",
        "formatting_normalized_hash",
        "exact_duplicate_group_id",
        "formatting_duplicate_group_id",
        "duplicate_group_id",
        "source_metadata",
        "source_key",
        "source_path",
        "source_row_index",
    ]
    df = df[ordered_columns]

    processed_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, group in df.groupby("dataset_name", sort=True):
        path = processed_dir / f"{dataset_name.lower()}_processed.csv"
        group.to_csv(path, index=False)
        logger.info("wrote %s rows=%d", path, len(group))
    pooled_path = processed_dir / "pooled_313_processed.csv"
    df.to_csv(pooled_path, index=False)
    df.to_json(processed_dir / "pooled_313_processed.jsonl", orient="records", lines=True, force_ascii=False)
    logger.info("wrote %s rows=%d", pooled_path, len(df))

    summary = (
        df.groupby("dataset_name")["human_mean_score"]
        .agg(["count", "min", "max", "mean", "std"])
        .reset_index()
    )
    summary["z_mean_check"] = df.groupby("dataset_name")["human_score_z_within_dataset"].mean().values
    summary["z_std_check"] = df.groupby("dataset_name")["human_score_z_within_dataset"].std(ddof=0).values
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "dataset_summary.csv", index=False)

    duplicate_report_rows = []
    for kind, col in [
        ("exact", "exact_duplicate_group_id"),
        ("formatting_only", "formatting_duplicate_group_id"),
    ]:
        dup = df[df[col] != ""]
        for group_id, group in dup.groupby(col):
            duplicate_report_rows.append(
                {
                    "duplicate_kind": kind,
                    "duplicate_group_id": group_id,
                    "n": len(group),
                    "rq0_ids": ",".join(group["rq0_id"].tolist()),
                    "datasets": ",".join(sorted(group["dataset_name"].unique())),
                    "snippet_ids": ",".join(map(str, group["snippet_id"].tolist())),
                }
            )
    duplicate_report = pd.DataFrame(duplicate_report_rows)
    duplicate_report.to_csv(tables_dir / "duplicate_report.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "raw_sources": config["raw_sources"],
        "row_count": int(len(df)),
        "dataset_counts": df["dataset_name"].value_counts().sort_index().to_dict(),
        "exact_duplicate_groups": int((df["exact_duplicate_group_id"] != "").sum()),
        "formatting_duplicate_groups": int((df["formatting_duplicate_group_id"] != "").sum()),
        "outputs": {
            "pooled_csv": str(pooled_path),
            "dataset_summary": str(tables_dir / "dataset_summary.csv"),
            "duplicate_report": str(tables_dir / "duplicate_report.csv"),
        },
    }
    write_json(processed_dir / "prepare_manifest.json", manifest)
    logger.info("dataset summary:\n%s", summary.to_string(index=False))
    logger.info("duplicate rows exact_or_formatting=%d", int((df["duplicate_group_id"] != "").sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

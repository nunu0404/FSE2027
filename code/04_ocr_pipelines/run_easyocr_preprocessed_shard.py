#!/usr/bin/env python3
"""Run or merge independent EasyOCR-preprocessed snippet shards."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def load_impl():
    path = ROOT / "experiments/rq0_viability/scripts/run_clean_multi_ocr_experiment.py"
    spec = importlib.util.spec_from_file_location("ocr_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_shard(index: int, count: int) -> None:
    impl = load_impl()
    main_results = ROOT / "results/python_cuda_vlm_main_20260715"
    ocr_root = ROOT / "results/python_cuda_missing_experiments_20260716/ocr"
    snippets = pd.read_csv(main_results / "data/snippets.csv")
    metadata = pd.read_csv(main_results / "render_metadata/default_render_metadata.csv")
    completed_path = ocr_root / "ocr_outputs/easyocr_preprocessed/snippet_ocr.csv"
    completed = pd.read_csv(completed_path) if completed_path.exists() else pd.DataFrame(columns=["snippet_id"])
    remaining = snippets[~snippets["rq0_id"].isin(set(completed["snippet_id"].astype(str)))].reset_index(drop=True)
    shard = remaining.iloc[index::count].copy()
    output = ocr_root / "preprocessed_shards" / f"shard_{index}"
    result = impl.run_easyocr_engine(shard, metadata, output, "easyocr_preprocessed", True, False)
    if len(result) != len(shard) or not result["parse_status"].eq("ok").all():
        raise RuntimeError(f"Shard {index} failed integrity: {result['parse_status'].value_counts().to_dict()}")
    print({"status": "passed", "shard": index, "rows": len(result)}, flush=True)


def merge_shards(count: int) -> None:
    ocr_root = ROOT / "results/python_cuda_missing_experiments_20260716/ocr"
    destination = ocr_root / "ocr_outputs/easyocr_preprocessed/snippet_ocr.csv"
    frames = [pd.read_csv(destination)]
    for index in range(count):
        path = ocr_root / f"preprocessed_shards/shard_{index}/ocr_outputs/easyocr_preprocessed/snippet_ocr.csv"
        frames.append(pd.read_csv(path))
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != 239 or merged["snippet_id"].nunique() != 239 or not merged["parse_status"].eq("ok").all():
        raise RuntimeError(
            f"Merged preprocessed integrity failed: rows={len(merged)} unique={merged['snippet_id'].nunique()} "
            f"statuses={merged['parse_status'].value_counts().to_dict()}"
        )
    merged.to_csv(destination, index=False)
    print({"status": "passed", "merged_rows": len(merged)}, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "merge"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    if args.command == "run":
        run_shard(args.index, args.count)
    else:
        merge_shards(args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

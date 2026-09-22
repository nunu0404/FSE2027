#!/usr/bin/env python3
"""Freeze references to the existing deployment inputs without regenerating them."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/deploy_v2_logit_20260731"
JAVA = ROOT / "results/screenshot_only_ocr_clean_20260707"
EXT = ROOT / "results/python_cuda_vlm_main_20260715"
MISS = ROOT / "results/python_cuda_missing_experiments_20260716"
RUNNER = ROOT / "experiments/rq0_viability/scripts/run_judge_pairs.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompt() -> str:
    spec = importlib.util.spec_from_file_location("f5_frozen_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PROMPTS["B"]


def main() -> None:
    data_dir = OUT / "data"
    config_dir = OUT / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    java_pairs_path = JAVA / "pair_image_mapping.csv"
    ext_pairs_path = EXT / "data/pairs_seed42_clean.csv"
    java_pairs = pd.read_csv(java_pairs_path)
    ext_pairs = pd.read_csv(ext_pairs_path)
    java_pairs["language"] = "java"
    java_pairs["source_run"] = "deploy_java_clean"
    ext_pairs["source_run"] = "deploy_python_cuda"
    columns = [
        "pair_id", "snippet_i", "snippet_j", "dataset_name_i", "dataset_name_j",
        "language", "difficulty", "human_score_i_z", "human_score_j_z",
        "abs_z_diff", "human_preference", "preference_score_basis", "source_run",
    ]
    pairs = pd.concat([java_pairs[columns], ext_pairs[columns]], ignore_index=True)
    counts = pairs.groupby("language").size().to_dict()
    if counts != {"cuda": 3000, "java": 3000, "python": 3000}:
        raise RuntimeError(f"pair allocation drift: {counts}")
    if pairs.pair_id.duplicated().any():
        raise RuntimeError("pair IDs are not globally unique")
    pairs.to_csv(data_dir / "F5_FROZEN_PAIRS.csv", index=False)

    java_source_path = ROOT / "experiments/rq0_viability/data/processed/pooled_313_processed.csv"
    java_ocr_path = JAVA / "llm_inputs/pooled_313_rapidocr_text.csv"
    ext_source_path = MISS / "ocr_text_llm_inputs/source_text.csv"
    ext_ocr_path = MISS / "ocr_text_llm_inputs/rapidocr_text.csv"
    java_source = pd.read_csv(java_source_path).set_index("rq0_id")
    java_ocr = pd.read_csv(java_ocr_path).set_index("rq0_id")
    ext_source = pd.read_csv(ext_source_path).set_index("rq0_id")
    ext_ocr = pd.read_csv(ext_ocr_path).set_index("rq0_id")
    java_images = java_pairs.set_index("snippet_i").image_i_path.to_dict()
    java_images.update(java_pairs.set_index("snippet_j").image_j_path.to_dict())
    ext_meta_path = EXT / "render_metadata/default_render_metadata.csv"
    ext_meta = pd.read_csv(ext_meta_path)
    ext_images = ext_meta.set_index("rq0_id").image_path.to_dict()

    item_rows = []
    for language, ids in pairs.groupby("language").apply(
        lambda group: sorted(set(group.snippet_i) | set(group.snippet_j)),
        include_groups=False,
    ).items():
        for snippet in ids:
            if language == "java":
                source, ocr, image = java_source, java_ocr, java_images[snippet]
            else:
                source, ocr, image = ext_source, ext_ocr, ext_images[snippet]
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            item_rows.append(
                {
                    "snippet_id": snippet,
                    "language": language,
                    "image_path": str(image_path),
                    "image_sha256": sha256(image_path),
                    "source_text": str(source.loc[snippet, "raw_code"]),
                    "rapidocr_text": str(ocr.loc[snippet, "raw_code"]),
                }
            )
    items = pd.DataFrame(item_rows)
    if len(items) != 552 or items.snippet_id.duplicated().any():
        raise RuntimeError(f"unexpected item population: {len(items)}")
    items.to_csv(data_dir / "F5_FROZEN_ITEMS.csv", index=False)

    prompt = load_prompt()
    prompt_path = config_dir / "F5_FROZEN_PROMPT_B.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    old_assets = [
        ROOT / "experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
        EXT / "outputs/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
        JAVA / "ocr_text_llm_raw/source_text_llm.jsonl",
        JAVA / "ocr_text_llm_raw/ocr_text_llm_rapidocr.jsonl",
        MISS / "ocr_text_llm_raw/source_text_llm.jsonl",
        MISS / "ocr_text_llm_raw/ocr_text_llm_rapidocr.jsonl",
    ]
    deterministic_assets = [
        JAVA / "ocr_classical_pair_predictions.csv",
        MISS / "ocr/ocr_classical_pair_predictions.csv",
    ]
    frozen_assets = [
        java_pairs_path,
        ext_pairs_path,
        java_source_path,
        java_ocr_path,
        ext_source_path,
        ext_ocr_path,
        ext_meta_path,
        RUNNER,
        prompt_path,
        *old_assets,
        *deterministic_assets,
    ]
    inventory = []
    for path in frozen_assets:
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory.append(
            {
                "asset_type": (
                    "old_output" if path in old_assets
                    else "deterministic_ocrml" if path in deterministic_assets
                    else "frozen_input"
                ),
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    for row in items.itertuples(index=False):
        inventory.append(
            {
                "asset_type": "frozen_png",
                "path": str(Path(row.image_path).relative_to(ROOT)),
                "bytes": Path(row.image_path).stat().st_size,
                "sha256": row.image_sha256,
            }
        )
    inventory_frame = pd.DataFrame(inventory).drop_duplicates("path")
    inventory_frame.to_csv(data_dir / "F5_FROZEN_INPUT_INVENTORY.csv", index=False)

    audit = {
        "run_id": "deploy_v2_logit_20260731",
        "gate": "frozen_input_audit",
        "gate_pass": True,
        "pairs": len(pairs),
        "pairs_by_language": counts,
        "unique_snippets": len(items),
        "png_assets": int((inventory_frame.asset_type == "frozen_png").sum()),
        "prompt_sha256": sha256(prompt_path),
        "pair_csv_sha256": sha256(data_dir / "F5_FROZEN_PAIRS.csv"),
        "item_csv_sha256": sha256(data_dir / "F5_FROZEN_ITEMS.csv"),
        "inventory_sha256": sha256(data_dir / "F5_FROZEN_INPUT_INVENTORY.csv"),
        "images_or_ocr_regenerated": False,
    }
    (data_dir / "F5_FROZEN_INPUT_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

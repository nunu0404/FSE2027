#!/usr/bin/env python3
"""Replicate the frozen deploy-v2 direct-visual route with Qwen3-VL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

import run_primary as primary


ROOT = primary.ROOT
OUT = primary.OUT
DEPLOY_ROOT = ROOT / "fse2027/external_runs/deploy_v2_logit_20260731"
PAIR_PATH = DEPLOY_ROOT / "data/F5_FROZEN_PAIRS.csv"
ITEM_PATH = DEPLOY_ROOT / "data/F5_FROZEN_ITEMS.csv"
PROMPT_PATH = OUT / "config/FROZEN_PROMPT_B.txt"
KST = timezone(timedelta(hours=9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(OUT), "rev-parse", "HEAD"], text=True
    ).strip()


def check_upstream_gates() -> None:
    for model in ("qwen3", "internvl3_5", "gemma4"):
        path = OUT / "inference/full" / model / "validation.json"
        if not path.is_file() or not json.loads(path.read_text()).get("gate_pass"):
            raise RuntimeError(f"primary validation is not passed: {model}")
    path = OUT / "inference/robustness/qwen3/validation.json"
    if not path.is_file() or not json.loads(path.read_text()).get("gate_pass"):
        raise RuntimeError("Qwen3 robustness validation is not passed")


def open_verified(path: str, expected: str) -> Image.Image:
    target = Path(path)
    if sha256(target) != expected:
        raise RuntimeError(f"deploy image hash drift: {target}")
    with Image.open(target) as image:
        return image.convert("RGB").copy()


def completed_keys(path: Path) -> set[tuple[str, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["pair_id"], row["order"])
            if key in keys:
                raise RuntimeError(f"duplicate output at line {number}: {key}")
            keys.add(key)
    return keys


def write_manifest(path: Path, manifest: dict, status: str, calls: int) -> None:
    manifest["status"] = status
    manifest["calls_completed"] = calls
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate(raw_path: Path, pairs: pd.DataFrame, expected_calls: int) -> dict:
    raw = pd.read_json(raw_path, lines=True)
    expected = pd.MultiIndex.from_product(
        [pairs.pair_id, ["AB", "BA"]], names=["pair_id", "order"]
    )
    actual = pd.MultiIndex.from_frame(raw[["pair_id", "order"]])
    counts = raw.groupby(["language", "order"]).size().to_dict()
    expected_counts = {
        (language, order): int((pairs.language == language).sum())
        for language in sorted(pairs.language.unique())
        for order in ("AB", "BA")
    }
    result = {
        "pair_count_from_manifest": len(pairs),
        "expected_calls_formula": "2 * pair_count_from_manifest",
        "expected_calls": expected_calls,
        "actual_calls": len(raw),
        "unique_pairs": raw.pair_id.nunique(),
        "duplicates": int(raw[["pair_id", "order"]].duplicated().sum()),
        "missing_keys": len(expected.difference(actual)),
        "unexpected_keys": len(actual.difference(expected)),
        "parse_failures": int(raw.parsed_choice.isna().sum()),
        "missing_logits": int((raw.logit_A.isna() | raw.logit_B.isna()).sum()),
        "argmax_mismatches": int((~raw.argmax_matches_parsed.fillna(False).astype(bool)).sum()),
        "mapping_errors": int((~raw.image_mapping_verified.astype(bool)).sum()),
        "language_order_complete": counts == expected_counts,
    }
    result["gate_pass"] = bool(
        result["actual_calls"] == expected_calls
        and result["unique_pairs"] == len(pairs)
        and result["language_order_complete"]
        and all(
            result[key] == 0
            for key in (
                "duplicates",
                "missing_keys",
                "unexpected_keys",
                "parse_failures",
                "missing_logits",
                "argmax_mismatches",
                "mapping_errors",
            )
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    check_upstream_gates()
    gpu = primary.physical_gpu()
    pairs = pd.read_csv(PAIR_PATH)
    items = pd.read_csv(ITEM_PATH).set_index("snippet_id")
    if len(pairs) != pairs.pair_id.nunique():
        raise RuntimeError("deploy-v2 pair manifest has duplicate pair IDs")
    expected_calls = 2 * len(pairs)
    counts = pairs.groupby("language").size().to_dict()
    if sum(counts.values()) != len(pairs):
        raise RuntimeError("deploy-v2 language count drift")
    config = primary.MODELS["qwen3"]
    snapshot = primary.snapshot_path(config["id"], config["revision"])
    run_dir = OUT / "inference/deployment/qwen3_direct_visual"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    manifest_path = run_dir / "manifest.json"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume")
    done = completed_keys(raw_path) if args.resume else set()
    manifest = {
        "run_family": "latest_vlm_extension_20260830",
        "phase": "deploy_v2_direct_visual_replication",
        "system": "direct_qwen3_image_only",
        "model_id": config["id"],
        "model_revision": config["revision"],
        "processor_revision": config["revision"],
        "package_git_commit": package_commit(),
        "runner_sha256": sha256(Path(__file__)),
        "source_pair_manifest": str(PAIR_PATH.relative_to(ROOT)),
        "source_pair_manifest_sha256": sha256(PAIR_PATH),
        "source_item_manifest": str(ITEM_PATH.relative_to(ROOT)),
        "source_item_manifest_sha256": sha256(ITEM_PATH),
        "prompt_sha256": sha256(PROMPT_PATH),
        "pair_count_from_manifest": len(pairs),
        "pairs_by_language": counts,
        "calls_expected_formula": "2 * pair_count_from_manifest",
        "calls_expected": expected_calls,
        "orders": ["AB", "BA"],
        "modality": "image_only",
        "packaging": "two_separate_ordered_deploy_v2_screenshots",
        "generation": json.loads((OUT / "config/protocol.json").read_text())[
            "generation"
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(manifest_path, manifest, "loading_model", len(done))
    adapter = primary.ModelAdapter("qwen3", snapshot)
    prompt_template = PROMPT_PATH.read_text().rstrip("\n")
    write_manifest(manifest_path, manifest, "running", len(done))
    start = time.monotonic()
    initial_done = len(done)
    formal_rows = []
    with raw_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for pair in pairs.itertuples(index=False):
            item_i = items.loc[pair.snippet_i]
            item_j = items.loc[pair.snippet_j]
            prompt = prompt_template.format(language=pair.language)
            for order, first_id, second_id, first, second in (
                ("AB", pair.snippet_i, pair.snippet_j, item_i, item_j),
                ("BA", pair.snippet_j, pair.snippet_i, item_j, item_i),
            ):
                key = (pair.pair_id, order)
                if key in done:
                    continue
                info = adapter.generate(
                    prompt,
                    open_verified(first.image_path, first.image_sha256),
                    open_verified(second.image_path, second.image_sha256),
                )
                mapping_verified = bool(
                    (order == "AB" and first_id == pair.snippet_i and second_id == pair.snippet_j)
                    or (order == "BA" and first_id == pair.snippet_j and second_id == pair.snippet_i)
                )
                row = {
                    "run_family": "latest_vlm_extension_20260830",
                    "phase": "deploy_v2_direct_visual_replication",
                    "system": "direct_qwen3_image_only",
                    "model": config["id"],
                    "model_revision": config["revision"],
                    "language": pair.language,
                    "difficulty": pair.difficulty,
                    "pair_id": pair.pair_id,
                    "abs_z_diff": pair.abs_z_diff,
                    "human_score_i_z": pair.human_score_i_z,
                    "human_score_j_z": pair.human_score_j_z,
                    "human_preference": pair.human_preference,
                    "snippet_i": pair.snippet_i,
                    "snippet_j": pair.snippet_j,
                    "snippet_first": first_id,
                    "snippet_second": second_id,
                    "image_first_path": first.image_path,
                    "image_second_path": second.image_path,
                    "image_first_sha256": first.image_sha256,
                    "image_second_sha256": second.image_sha256,
                    "image_mapping_verified": mapping_verified,
                    "order": order,
                    "gold_side": "first" if pair.human_preference == first_id else "second",
                    **info,
                    "seed": 42,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                done.add(key)
                if len(formal_rows) < 20:
                    formal_rows.append(row)
                    if len(formal_rows) == 20:
                        gate = pd.DataFrame(formal_rows)
                        if not (
                            gate.parsed_choice.notna().all()
                            and gate.logit_A.notna().all()
                            and gate.logit_B.notna().all()
                            and gate.argmax_matches_parsed.astype(bool).all()
                            and gate.image_mapping_verified.astype(bool).all()
                        ):
                            write_manifest(
                                manifest_path, manifest, "formal_readiness_failed", len(done)
                            )
                            raise RuntimeError("deployment formal readiness gate failed")
                if len(done) % 100 == 0:
                    elapsed = time.monotonic() - start
                    rate = (len(done) - initial_done) / elapsed if elapsed else 0
                    remaining = expected_calls - len(done)
                    eta = datetime.now(KST) + timedelta(seconds=remaining / rate) if rate else None
                    print(
                        f"PROGRESS system=direct_qwen3_image_only calls={len(done)}/{expected_calls} "
                        f"rate={rate:.3f}_calls_s eta_kst={eta.strftime('%F %T') if eta else 'NA'}",
                        flush=True,
                    )
                if len(done) % 500 == 0:
                    write_manifest(manifest_path, manifest, "running", len(done))

    result = validate(raw_path, pairs, expected_calls)
    (run_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(
        manifest_path,
        manifest,
        "validated_complete" if result["gate_pass"] else "validation_failed",
        len(done),
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

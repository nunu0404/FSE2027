#!/usr/bin/env python3
"""Validated A-grid/B-perturbation VLM runner for the grounded protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
PILOT_PATH = ROOT / "results/protocol_unified_3lang_20260720/code/run_logit_pilot.py"
MODEL_NAMES = ["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pilot():
    spec = importlib.util.spec_from_file_location("grounded_logit_capture", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def open_rgb(relative_path: str) -> Image.Image:
    with Image.open(ROOT / relative_path) as image:
        return image.convert("RGB").copy()


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["condition"], row["pair_id"], row["order"])
            if key in keys:
                raise ValueError(f"duplicate key at {path}:{line_number}: {key}")
            keys.add(key)
    return keys


def model_snapshot(model: str, revision: str) -> Path:
    org, name = model.split("/", 1)
    return Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "snapshots" / revision


def model_main_ref(model: str) -> Path:
    org, name = model.split("/", 1)
    return Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "refs/main"


def build_plan(experiment: str) -> tuple[pd.DataFrame, list[str], dict[tuple[str, str], str]]:
    pairs = pd.read_csv(OUT / "data/pairs_seed42_grounded.csv")
    if len(pairs) != 3000 or pairs.groupby("language").size().to_dict() != {"cuda": 1000, "java": 1000, "python": 1000}:
        raise ValueError("pair allocation drift")
    if pairs[["protocol_pair_id", "snippet_i", "snippet_j", "human_preference"]].isna().any().any():
        raise ValueError("required pair value is missing")
    if experiment == "grid":
        metadata_path = OUT / "rendered/metadata/grid_render_metadata.csv"
        conditions = pd.read_csv(OUT / "data/render_conditions.csv").condition.tolist()
        baseline = "monokai_dark__fs20__wrap80__lnon"
        conditions = [baseline] + [value for value in conditions if value != baseline]
        expected_conditions = 12
    else:
        metadata_path = OUT / "rendered/metadata/perturbation_render_metadata.csv"
        conditions = ["baseline", "no_indent", "no_blank_lines", "gaussian_sigma_1", "gaussian_sigma_2", "gaussian_sigma_4"]
        expected_conditions = 6
    if len(conditions) != expected_conditions or len(set(conditions)) != expected_conditions:
        raise ValueError("condition allocation drift")
    metadata = pd.read_csv(metadata_path)
    lookup = metadata.set_index(["condition", "rq0_id"]).image_path.to_dict()
    snippets = set(pairs.snippet_i) | set(pairs.snippet_j)
    missing = [(condition, snippet) for condition in conditions for snippet in snippets if (condition, snippet) not in lookup]
    if missing:
        raise ValueError(f"missing rendered inputs: {missing[:5]} ({len(missing)} total)")
    paths = {lookup[(condition, snippet)] for condition in conditions for snippet in snippets}
    absent = [path for path in paths if not (ROOT / path).is_file()]
    if absent:
        raise FileNotFoundError(f"missing image files: {absent[:5]} ({len(absent)} total)")
    return pairs, conditions, lookup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["grid", "perturbation"])
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-calls", type=int, default=0, help="Pilot only; zero runs the complete plan.")
    parser.add_argument("--output-tag", default="full", help="Use a distinct tag for every pilot.")
    args = parser.parse_args()

    protocol_path = OUT / "config/protocol_grounded.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prompt_path = ROOT / protocol["inference"]["prompt"]
    prompt_template = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    expected = {"dtype": "bfloat16", "do_sample": False, "temperature": 0.0, "top_p": 1.0,
                "top_p_active": False, "max_new_tokens": 24, "seed": 42, "modality": "image_only",
                "direct_pairwise": True, "strict_swap": True, "capture_verdict_logits": True}
    for key, value in expected.items():
        if protocol["inference"].get(key) != value:
            raise ValueError(f"protocol drift for {key}: {protocol['inference'].get(key)!r} != {value!r}")
    revision = protocol["inference"]["model_revisions"][args.model]
    snapshot = model_snapshot(args.model, revision)
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned model snapshot is absent: {snapshot}")
    main_ref = model_main_ref(args.model)
    if not main_ref.is_file() or main_ref.read_text(encoding="utf-8").strip() != revision:
        raise RuntimeError(f"local main ref does not match pinned revision: {main_ref}")
    render_audit = json.loads((OUT / "audit/GROUNDED_RENDER_AUDIT.json").read_text())
    if not render_audit.get("gate_pass"):
        raise RuntimeError("render audit gate did not pass")

    pairs, conditions, image_lookup = build_plan(args.experiment)
    pilot = load_pilot()
    runner = pilot.load_runner()
    if runner.PROMPTS["B"] != prompt_template:
        raise RuntimeError("frozen Prompt B differs from the inference implementation")
    expected_calls = len(pairs) * len(conditions) * 2
    if args.limit_calls < 0 or args.limit_calls > expected_calls:
        raise ValueError("invalid --limit-calls")
    planned_calls = args.limit_calls or expected_calls
    safe_model = args.model.replace("/", "__")
    run_dir = OUT / "inference" / args.experiment / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_model}__image_only__promptB__seed42__{args.output_tag}"
    raw_path, manifest_path = run_dir / f"{stem}.jsonl", run_dir / f"{stem}.manifest.json"
    done = completed_keys(raw_path) if args.resume else set()
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume or a new --output-tag: {raw_path}")

    manifest = {
        "status": "dry_run_passed" if args.dry_run else "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": protocol["protocol_version"], "protocol_sha256": sha256(protocol_path),
        "render_audit_sha256": sha256(OUT / "audit/GROUNDED_RENDER_AUDIT.json"),
        "experiment": args.experiment, "model": args.model, "model_revision": revision,
        "model_snapshot": str(snapshot), "prompt_path": str(prompt_path.relative_to(ROOT)),
        "prompt_sha256": sha256(prompt_path), "prompt_template": prompt_template,
        "languages": ["java", "python", "cuda"], "pairs": len(pairs), "conditions": conditions,
        "orders": ["AB", "BA"], "expected_full_calls": expected_calls, "planned_calls": planned_calls,
        "modality": "image_only", "packaging": "two_separately_labeled_images", "prompt_variant": "B",
        "dtype": "bfloat16", "decoding": {"do_sample": False, "temperature": 0.0,
        "top_p_inactive": 1.0, "max_new_tokens": 24}, "seed": 42,
        "capture_verdict_logits": True, "resume_completed_calls": len(done),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "python": platform.python_version(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    runner.set_global_seed(42)
    if "Qwen" in args.model:
        judge = runner.DirectQwenVLJudge(args.model, "auto", "bf16", 24, 0.0, 1.0)
        generate = pilot.qwen_generate
    else:
        judge = runner.DirectInternVLJudge(args.model, "bf16", 24, 0.0, 1.0)
        generate = pilot.internvl_generate

    written = 0
    with raw_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for condition in conditions:
            for pair in pairs.itertuples(index=False):
                prompt = runner.build_prompt("B", None, None, False, pair.language)
                for order, first_id, second_id in (("AB", pair.snippet_i, pair.snippet_j), ("BA", pair.snippet_j, pair.snippet_i)):
                    key = (condition, pair.protocol_pair_id, order)
                    if key in done:
                        continue
                    if len(done) + written >= planned_calls:
                        break
                    _, info = generate(judge, prompt, open_rgb(image_lookup[(condition, first_id)]),
                                       open_rgb(image_lookup[(condition, second_id)]))
                    row = {"experiment": args.experiment, "condition": condition, "model": args.model,
                           "model_revision": revision, "language": pair.language, "pair_id": pair.protocol_pair_id,
                           "source_pair_id": pair.pair_id, "difficulty_rank": pair.difficulty_rank,
                           "abs_z_diff": pair.abs_z_diff, "snippet_i": pair.snippet_i, "snippet_j": pair.snippet_j,
                           "snippet_first": first_id, "snippet_second": second_id, "order": order,
                           "gold_side": "first" if pair.human_preference == first_id else "second", **info,
                           "seed": 42, "created_at_utc": datetime.now(timezone.utc).isoformat()}
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    handle.flush()
                    written += 1
                    if written % 25 == 0:
                        print(f"{args.experiment} {args.model} calls={len(done)+written}/{planned_calls}", flush=True)
                if len(done) + written >= planned_calls:
                    break
            if len(done) + written >= planned_calls:
                break
    manifest["status"] = "complete" if len(done) + written == expected_calls else "pilot_complete"
    manifest["completed_calls"] = len(done) + written
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "calls": manifest["completed_calls"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run author-gated direct-pairwise RQ3 inference with verdict logits."""

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
RQ3 = OUT / "rq3"
PILOT_PATH = ROOT / "results/protocol_unified_3lang_20260720/code/run_logit_pilot.py"
MODELS = ["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B"]
MODALITIES = ["image_only", "text_plus_image"]
REVISIONS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "OpenGVLab/InternVL3-8B": "853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pilot():
    spec = importlib.util.spec_from_file_location("rq3_logit_capture", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(model: str, revision: str) -> Path:
    org, name = model.split("/", 1)
    return Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "snapshots" / revision


def open_rgb(relative_path: str) -> Image.Image:
    with Image.open(ROOT / relative_path) as image:
        return image.convert("RGB").copy()


def completed_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["contrast_id"], row["order"])
            if key in keys:
                raise ValueError(f"duplicate key at line {line_number}: {key}")
            keys.add(key)
    return keys


def load_plan():
    contrasts = pd.read_csv(RQ3 / "data/rq3_contrast_manifest.csv")
    sources = pd.read_csv(RQ3 / "data/rq3_variant_sources.csv")
    metadata = pd.read_csv(RQ3 / "data/rq3_render_metadata.csv")
    if len(contrasts) != 1800 or contrasts.contrast_id.nunique() != 1800:
        raise ValueError("RQ3 contrast allocation drift")
    if contrasts.groupby("language").size().to_dict() != {"cuda": 600, "java": 600, "python": 600}:
        raise ValueError("RQ3 language allocation drift")
    if len(metadata) != 1200 or metadata.duplicated(["base_id", "variant"]).any():
        raise ValueError("RQ3 render metadata drift")
    images = metadata.set_index(["base_id", "variant"]).image_path.to_dict()
    semantics = metadata.set_index(["base_id", "variant"]).semantic_integrity.to_dict()
    source_lookup = sources.set_index(["base_id", "semantic_integrity"]).code.to_dict()
    absent = [value for value in set(images.values()) if not (ROOT / value).is_file()]
    if absent:
        raise FileNotFoundError(f"missing RQ3 images: {absent[:5]}")
    return contrasts, images, semantics, source_lookup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--modality", required=True, choices=MODALITIES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write a plan without loading a model; GC1 may still be pending.")
    parser.add_argument("--limit-calls", type=int, default=0, help="Non-primary pilot only; zero means all 3,600 calls.")
    parser.add_argument("--output-tag", default="full_20260721")
    args = parser.parse_args()

    automatic_path = RQ3 / "audit/RQ3_AUTOMATIC_AUDIT.json"
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    if not automatic.get("automatic_gate_pass"):
        raise RuntimeError("RQ3 automatic preparation gate did not pass")
    approval_path = RQ3 / "review/GC1_APPROVAL.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8")) if approval_path.exists() else None
    gc1_pass = bool(approval and approval.get("gate_pass"))
    if not args.dry_run and not gc1_pass:
        raise RuntimeError("RQ3 inference blocked: complete GC1_REVIEW_FORM.csv and run validate_rq3_gc1.py")

    revision = REVISIONS[args.model]
    model_snapshot = snapshot(args.model, revision)
    if not model_snapshot.is_dir():
        raise FileNotFoundError(f"pinned model snapshot is absent: {model_snapshot}")
    contrasts, images, semantics, source_lookup = load_plan()
    expected_calls = len(contrasts) * 2
    if args.limit_calls < 0 or args.limit_calls > expected_calls:
        raise ValueError("invalid --limit-calls")
    planned_calls = args.limit_calls or expected_calls
    prompt_path = OUT / "config/prompt_B_template.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    pilot = load_pilot()
    runner = pilot.load_runner()
    if runner.PROMPTS["B"] != prompt_template:
        raise RuntimeError("frozen Prompt B differs from the inference implementation")

    safe_model = args.model.replace("/", "__")
    raw_dir = RQ3 / "inference/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_model}__{args.modality}__promptB__seed42__{args.output_tag}"
    raw_path = raw_dir / f"{stem}.jsonl"
    manifest_path = raw_dir / f"{stem}.manifest.json"
    done = completed_keys(raw_path) if args.resume else set()
    if raw_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"output exists; use --resume or another --output-tag: {raw_path}")
    manifest = {
        "status": "dry_run_passed" if args.dry_run else "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "C_RQ3", "model": args.model, "model_revision": revision,
        "model_snapshot": str(model_snapshot), "modality": args.modality,
        "packaging": "two_separately_labeled_images", "prompt_variant": "B",
        "prompt_path": str(prompt_path.relative_to(ROOT)), "prompt_sha256": sha256(prompt_path),
        "languages": ["java", "python", "cuda"], "bases": 300, "contrasts": len(contrasts),
        "contrasts_by_language": contrasts.groupby("language").size().to_dict(),
        "orders": ["AB", "BA"], "expected_full_calls": expected_calls, "planned_calls": planned_calls,
        "dtype": "bfloat16", "decoding": {"do_sample": False, "temperature": 0.0,
        "top_p_inactive": 1.0, "max_new_tokens": 24}, "seed": 42,
        "capture_verdict_logits": True, "strict_swap": True,
        "automatic_audit_sha256": sha256(automatic_path), "gc1_authorized": gc1_pass,
        "gc1_approval_sha256": sha256(approval_path) if gc1_pass else None,
        "resume_completed_calls": len(done), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
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
        for pair in contrasts.itertuples(index=False):
            for order, first_variant, second_variant in (
                ("AB", pair.variant_i, pair.variant_j),
                ("BA", pair.variant_j, pair.variant_i),
            ):
                key = (pair.contrast_id, order)
                if key in done:
                    continue
                if len(done) + written >= planned_calls:
                    break
                first_code = source_lookup[(pair.base_id, semantics[(pair.base_id, first_variant)])]
                second_code = source_lookup[(pair.base_id, semantics[(pair.base_id, second_variant)])]
                prompt = runner.build_prompt(
                    "B", first_code, second_code, args.modality == "text_plus_image", pair.language
                )
                _, info = generate(
                    judge,
                    prompt,
                    open_rgb(images[(pair.base_id, first_variant)]),
                    open_rgb(images[(pair.base_id, second_variant)]),
                )
                row = {
                    "exp": "C_RQ3", "language": pair.language, "model": args.model,
                    "model_revision": revision, "modality": args.modality,
                    "contrast_id": pair.contrast_id, "contrast_type": pair.contrast_type,
                    "base_id": pair.base_id, "rq0_id": pair.rq0_id, "score_stratum": pair.score_stratum,
                    "variant_i": pair.variant_i, "variant_j": pair.variant_j,
                    "preference_target_variant": pair.preference_target_variant,
                    "first_variant": first_variant, "second_variant": second_variant, "order": order,
                    "target_side": "first" if pair.preference_target_variant == first_variant else "second",
                    **info, "seed": 42, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                written += 1
                if written % 25 == 0:
                    print(f"RQ3 {args.model} {args.modality} calls={len(done)+written}/{planned_calls}", flush=True)
            if len(done) + written >= planned_calls:
                break
    total = len(done) + written
    manifest["status"] = "complete" if total == expected_calls else "pilot_complete"
    manifest["completed_calls"] = total
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "calls": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the four-condition presentation/modality robustness extension."""

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
PAIR_PATH = OUT / "data/robustness_pairs_900_seed42.csv"
COMBINED_META_PATH = OUT / "data/robustness_combined_metadata.csv"
SOURCE_PATH = ROOT / "results/grounded_protocol_3lang_20260721/data/render_snippets.csv"
PROMPT_PATH = OUT / "config/FROZEN_PROMPT_B.txt"
KST = timezone(timedelta(hours=9))
CONDITIONS = (
    "separate_image_only",
    "combined_image_only",
    "separate_text_plus_image",
    "combined_text_plus_image",
)


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


def check_primary_gates() -> None:
    for model in ("qwen3", "internvl3_5", "gemma4"):
        path = OUT / "inference/full" / model / "validation.json"
        if not path.is_file() or not json.loads(path.read_text()).get("gate_pass"):
            raise RuntimeError(f"primary validation is not passed: {model}")


def open_image(path: str, expected_hash: str | None = None) -> Image.Image:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    if expected_hash is not None and sha256(target) != expected_hash:
        raise RuntimeError(f"image hash drift: {target}")
    with Image.open(target) as image:
        return image.convert("RGB").copy()


def build_prompt(
    base: str,
    language: str,
    source_a: str,
    source_b: str,
    include_text: bool,
    combined: bool,
) -> str:
    prompt = base.format(language=language)
    if include_text:
        fence = {"java": "java", "python": "python", "cuda": "cuda"}[language]
        prompt += (
            f"\n\nCode A:\n```{fence}\n{source_a}\n```"
            f"\n\nCode B:\n```{fence}\n{source_b}\n```"
        )
    if combined:
        prompt += (
            "\n\nThe provided image is a single combined canvas. "
            "It contains two panels explicitly labeled Code A and Code B."
        )
    return prompt


def generate(
    adapter: primary.ModelAdapter,
    prompt: str,
    images: list[Image.Image],
    combined: bool,
) -> dict:
    content = [{"type": "text", "text": prompt}]
    if combined:
        content.extend(
            [
                {"type": "text", "text": "Code A image:"},
                {"type": "image", "image": images[0]},
            ]
        )
    else:
        content.extend(
            [
                {"type": "text", "text": "Code A image:"},
                {"type": "image", "image": images[0]},
                {"type": "text", "text": "Code B image:"},
                {"type": "image", "image": images[1]},
            ]
        )
    messages = [{"role": "user", "content": content}]
    inputs = adapter.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = getattr(adapter.model, "device", next(adapter.model.parameters()).device)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    hook, captured = primary.capture_lm_head_scores(adapter.model, adapter.tokenizer)
    try:
        with adapter.torch.inference_mode():
            output = adapter.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                return_dict_in_generate=True,
            )
    finally:
        hook.remove()
    input_length = inputs["input_ids"].shape[-1]
    token_ids = output.sequences[0, input_length:].tolist()
    raw = adapter.tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    info = primary.extract_hook_logits(adapter.tokenizer, token_ids, captured)
    info.update(
        {
            "raw_output": raw,
            "gen_tokens": len(token_ids),
            "reached_max_new_tokens": len(token_ids) >= 24,
        }
    )
    return info


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["condition"], row["pair_id"], row["order"])
            if key in keys:
                raise RuntimeError(f"duplicate key at line {number}: {key}")
            keys.add(key)
    return keys


def write_manifest(path: Path, manifest: dict, status: str, calls: int) -> None:
    manifest["status"] = status
    manifest["calls_completed"] = calls
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate(raw_path: Path) -> dict:
    raw = pd.read_json(raw_path, lines=True)
    pairs = pd.read_csv(PAIR_PATH)
    expected = pd.MultiIndex.from_product(
        [CONDITIONS, pairs.protocol_pair_id, ["AB", "BA"]],
        names=["condition", "pair_id", "order"],
    )
    actual = pd.MultiIndex.from_frame(raw[["condition", "pair_id", "order"]])
    counts = raw.groupby(["condition", "language", "order"]).size().to_dict()
    expected_counts = {
        (condition, language, order): 300
        for condition in CONDITIONS
        for language in ("java", "python", "cuda")
        for order in ("AB", "BA")
    }
    result = {
        "expected_calls": 7200,
        "actual_calls": len(raw),
        "unique_pairs": raw.pair_id.nunique(),
        "duplicates": int(raw[["condition", "pair_id", "order"]].duplicated().sum()),
        "missing_keys": len(expected.difference(actual)),
        "unexpected_keys": len(actual.difference(expected)),
        "parse_failures": int(raw.parsed_choice.isna().sum()),
        "missing_logits": int((raw.logit_A.isna() | raw.logit_B.isna()).sum()),
        "argmax_mismatches": int((~raw.argmax_matches_parsed.fillna(False).astype(bool)).sum()),
        "mapping_errors": int((~raw.image_mapping_verified.astype(bool)).sum()),
        "condition_language_order_complete": counts == expected_counts,
    }
    result["gate_pass"] = bool(
        result["actual_calls"] == 7200
        and result["unique_pairs"] == 900
        and result["condition_language_order_complete"]
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
    parser.add_argument("--model", required=True, choices=["qwen3", "internvl3_5"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    check_primary_gates()
    gpu = primary.physical_gpu()
    pairs = pd.read_csv(PAIR_PATH)
    combined_meta = pd.read_csv(COMBINED_META_PATH).set_index(["pair_id", "order"])
    if len(pairs) != 900 or len(combined_meta) != 1800:
        raise RuntimeError("robustness asset count drift")
    model_config = primary.MODELS[args.model]
    snapshot = primary.snapshot_path(model_config["id"], model_config["revision"])
    run_dir = OUT / "inference/robustness" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    manifest_path = run_dir / "manifest.json"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume")
    done = completed_keys(raw_path) if args.resume else set()
    manifest = {
        "run_family": "latest_vlm_extension_20260830",
        "phase": "robustness",
        "model_key": args.model,
        "model_id": model_config["id"],
        "model_revision": model_config["revision"],
        "processor_revision": model_config["revision"],
        "package_git_commit": package_commit(),
        "runner_sha256": sha256(Path(__file__)),
        "pair_manifest_sha256": sha256(PAIR_PATH),
        "combined_metadata_sha256": sha256(COMBINED_META_PATH),
        "source_text_sha256": sha256(SOURCE_PATH),
        "prompt_sha256": sha256(PROMPT_PATH),
        "conditions": list(CONDITIONS),
        "pairs": 900,
        "calls_expected": 7200,
        "orders": ["AB", "BA"],
        "generation": json.loads((OUT / "config/protocol.json").read_text())[
            "generation"
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(manifest_path, manifest, "loading_model", len(done))
    adapter = primary.ModelAdapter(args.model, snapshot)
    base_prompt = PROMPT_PATH.read_text().rstrip("\n")
    write_manifest(manifest_path, manifest, "running", len(done))
    start = time.monotonic()
    initial_done = len(done)
    formal_gate_rows = []
    with raw_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for pair in pairs.itertuples(index=False):
            for condition in CONDITIONS:
                include_text = "text_plus_image" in condition
                combined = condition.startswith("combined")
                for order in ("AB", "BA"):
                    key = (condition, pair.protocol_pair_id, order)
                    if key in done:
                        continue
                    if order == "AB":
                        first_id, second_id = pair.snippet_i, pair.snippet_j
                        source_a, source_b = pair.source_i, pair.source_j
                        path_a, path_b = pair.image_i_path, pair.image_j_path
                        hash_a, hash_b = pair.image_i_sha256, pair.image_j_sha256
                    else:
                        first_id, second_id = pair.snippet_j, pair.snippet_i
                        source_a, source_b = pair.source_j, pair.source_i
                        path_a, path_b = pair.image_j_path, pair.image_i_path
                        hash_a, hash_b = pair.image_j_sha256, pair.image_i_sha256
                    prompt = build_prompt(
                        base_prompt,
                        pair.language,
                        source_a,
                        source_b,
                        include_text,
                        combined,
                    )
                    if combined:
                        combined_row = combined_meta.loc[(pair.protocol_pair_id, order)]
                        images = [
                            open_image(combined_row.image_path, combined_row.image_sha256)
                        ]
                        input_paths = [combined_row.image_path]
                    else:
                        images = [open_image(path_a, hash_a), open_image(path_b, hash_b)]
                        input_paths = [path_a, path_b]
                    info = generate(adapter, prompt, images, combined)
                    mapping_verified = bool(
                        (order == "AB" and first_id == pair.snippet_i and second_id == pair.snippet_j)
                        or (order == "BA" and first_id == pair.snippet_j and second_id == pair.snippet_i)
                    )
                    if combined:
                        mapping_verified = bool(
                            mapping_verified
                            and combined_row.snippet_first == first_id
                            and combined_row.snippet_second == second_id
                        )
                    row = {
                        "run_family": "latest_vlm_extension_20260830",
                        "phase": "robustness",
                        "model": model_config["id"],
                        "model_revision": model_config["revision"],
                        "condition": condition,
                        "language": pair.language,
                        "difficulty": pair.difficulty,
                        "pair_id": pair.protocol_pair_id,
                        "abs_z_diff": pair.abs_z_diff,
                        "human_preference": pair.human_preference,
                        "snippet_i": pair.snippet_i,
                        "snippet_j": pair.snippet_j,
                        "snippet_first": first_id,
                        "snippet_second": second_id,
                        "image_input_paths": input_paths,
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
                    if len(formal_gate_rows) < 8:
                        formal_gate_rows.append(row)
                        if len(formal_gate_rows) == 8:
                            gate = pd.DataFrame(formal_gate_rows)
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
                                raise RuntimeError("formal robustness readiness gate failed")
                    if len(done) % 100 == 0:
                        elapsed = time.monotonic() - start
                        rate = (len(done) - initial_done) / elapsed if elapsed else 0
                        remaining = 7200 - len(done)
                        eta = datetime.now(KST) + timedelta(seconds=remaining / rate) if rate else None
                        print(
                            f"PROGRESS model={args.model} phase=robustness "
                            f"calls={len(done)}/7200 rate={rate:.3f}_calls_s "
                            f"eta_kst={eta.strftime('%F %T') if eta else 'NA'}",
                            flush=True,
                        )
                    if len(done) % 500 == 0:
                        write_manifest(manifest_path, manifest, "running", len(done))

    result = validate(raw_path)
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

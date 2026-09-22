#!/usr/bin/env python3
"""Run Gemma-3-12b-it (Gemma4 requested) on RQ2 reduced grid (7 conditions, 42,000 calls) + Qwen Anchor (6,000 calls)."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = Path("/ANON/experiment_root")
BASE_EXP = ROOT / "results/grounded_protocol_3lang_20260721"
OUT_DIR = ROOT / "rq2_eval_reduced"

GEMMA_SNAPSHOT = Path("/ANON/scratch_rq1/hf/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80")
GEMMA_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"

QWEN_SNAPSHOT = Path("/ANON/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"

VERDICT_RE = re.compile(r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B)\b", re.IGNORECASE)

CONDITIONS = [
    # 4 Rendering conditions
    ("grid", "monokai_dark__fs20__wrap80__lnon"),
    ("grid", "monokai_dark__fs20__wrap60__lnon"),
    ("grid", "monokai_dark__fs24__wrap80__lnon"),
    ("grid", "mono_light__fs20__wrap80__lnon"),
    # 3 Perturbation conditions
    ("perturbation", "no_indent"),
    ("perturbation", "no_blank_lines"),
    ("perturbation", "gaussian_sigma_4"),
]

PROMPT_TEMPLATE = """You are comparing two {language} code snippets by readability only.

Readability means how easily a developer can understand the local structure and intent of the code.

Consider:
1. Visual clarity: indentation consistency, spacing, line breaks, density.
2. Structural readability: block separation and control-flow traceability.
3. Information efficiency: naming clarity and local comprehensibility when text is legible.

Ignore functional correctness unless the code is locally corrupted or uninterpretable.

Return exactly one line:
FINAL_VERDICT: A
or
FINAL_VERDICT: B"""

DISPLAY_LANG = {"java": "Java", "python": "Python", "cuda": "CUDA"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_lookup() -> dict[tuple[str, str], str]:
    lookup = {}
    grid_meta = pd.read_csv(BASE_EXP / "rendered/metadata/grid_render_metadata.csv")
    for row in grid_meta.itertuples():
        lookup[(row.condition, row.rq0_id)] = row.image_path
        
    pert_meta = pd.read_csv(BASE_EXP / "rendered/metadata/perturbation_render_metadata.csv")
    for row in pert_meta.itertuples():
        lookup[(row.condition, row.rq0_id)] = row.image_path
    return lookup


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            keys.add((r["condition"], r["pair_id"], r["order"]))
    return keys


def extract_logits(tokenizer, token_ids: list[int], scores: list[Any], cand_a_id: int, cand_b_id: int) -> dict[str, Any]:
    verdict_index = None
    parsed = None
    for i in range(len(token_ids)):
        sub = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        m = VERDICT_RE.search(sub)
        if m:
            verdict_index = i
            parsed = m.group(1).upper()
            break

    if verdict_index is None:
        # Fallback to whole text search
        full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        m = VERDICT_RE.search(full_text)
        if m:
            parsed = m.group(1).upper()
        return {
            "parsed_choice": parsed,
            "verdict_token_index": None,
            "verdict_token_id": None,
            "token_family": "space_prefixed",
            "token_id_A": cand_a_id,
            "token_id_B": cand_b_id,
            "logit_A": None,
            "logit_B": None,
            "margin": None,
            "logit_tie": None,
            "logit_argmax_choice": None,
            "argmax_matches_parsed": False,
        }

    score_step = scores[verdict_index][0]
    logit_a = float(score_step[cand_a_id])
    logit_b = float(score_step[cand_b_id])
    margin = logit_a - logit_b
    argmax_choice = "A" if margin > 0 else ("B" if margin < 0 else "TIE")
    matches = (argmax_choice == parsed)

    return {
        "parsed_choice": parsed,
        "verdict_token_index": verdict_index,
        "verdict_token_id": token_ids[verdict_index],
        "token_family": "space_prefixed",
        "token_id_A": cand_a_id,
        "token_id_B": cand_b_id,
        "logit_A": round(logit_a, 4),
        "logit_B": round(logit_b, 4),
        "margin": round(margin, 4),
        "logit_tie": (margin == 0.0),
        "logit_argmax_choice": argmax_choice,
        "argmax_matches_parsed": matches,
    }


def run_gemma(pairs: pd.DataFrame, lookup: dict[tuple[str, str], str], device: int = 0) -> None:
    print(f"\n=======================================================")
    print(f"[GEMMA-RUN] Initializing google/gemma-3-12b-it on cuda:{device}...")
    print(f"=======================================================")
    
    out_raw = OUT_DIR / "gemma/raw_calls/gemma4_12b_reduced_raw.jsonl"
    out_manifest = OUT_DIR / "manifests/gemma4_12b_reduced_manifest.json"
    done = completed_keys(out_raw)
    total_calls = len(CONDITIONS) * len(pairs) * 2  # 7 * 3000 * 2 = 42,000
    print(f"[GEMMA-RUN] Total planned calls: {total_calls} | Already done: {len(done)}")

    if len(done) >= total_calls:
        print("[GEMMA-RUN] All Gemma calls already completed!")
        return

    processor = AutoProcessor.from_pretrained(GEMMA_SNAPSHOT, local_files_only=True)
    tokenizer = processor.tokenizer
    cand_a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    cand_b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    print(f"[GEMMA-RUN] Candidate tokens: ' A'={cand_a_id}, ' B'={cand_b_id}")

    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_SNAPSHOT,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    print("[GEMMA-RUN] Model loaded on GPU.")

    images_cache: dict[str, Image.Image] = {}

    def get_image(rel_path: str) -> Image.Image:
        if rel_path not in images_cache:
            if len(images_cache) > 200:
                images_cache.clear()
            with Image.open(ROOT / rel_path) as img:
                images_cache[rel_path] = img.convert("RGB").copy()
        return images_cache[rel_path]

    written = 0
    start_time = time.time()
    mismatch_count = 0

    with out_raw.open("a", encoding="utf-8") as handle:
        for exp_type, cond in CONDITIONS:
            print(f"\n[GEMMA-RUN] Starting condition: {cond} ({exp_type})")
            for pair in pairs.itertuples(index=False):
                prompt = PROMPT_TEMPLATE.format(language=DISPLAY_LANG[pair.language])
                for order, first_id, second_id in (("AB", pair.snippet_i, pair.snippet_j), ("BA", pair.snippet_j, pair.snippet_i)):
                    key = (cond, pair.protocol_pair_id, order)
                    if key in done:
                        continue

                    img_first = get_image(lookup[(cond, first_id)])
                    img_second = get_image(lookup[(cond, second_id)])

                    content = [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "Code A image:"},
                        {"type": "image", "image": img_first},
                        {"type": "text", "text": "Code B image:"},
                        {"type": "image", "image": img_second},
                    ]
                    messages = [{"role": "user", "content": content}]
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=24,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )

                    input_length = inputs["input_ids"].shape[1]
                    token_ids = output.sequences[0, input_length:].tolist()
                    raw_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

                    info = extract_logits(tokenizer, token_ids, list(output.scores), cand_a_id, cand_b_id)
                    info.update({"raw_output": raw_text, "gen_tokens": len(token_ids), "capture_method": "generate_output_scores"})

                    if not info["argmax_matches_parsed"]:
                        mismatch_count += 1
                        print(f"[WARNING] Argmax mismatch: parsed={info['parsed_choice']}, argmax={info['logit_argmax_choice']}, raw={raw_text!r}", flush=True)

                    gold_side = "first" if pair.human_preference == first_id else "second"
                    row = {
                        "experiment": exp_type,
                        "condition": cond,
                        "model": "google/gemma-3-12b-it",
                        "model_label": "Gemma4-12B",
                        "model_revision": GEMMA_REVISION,
                        "language": pair.language,
                        "pair_id": pair.protocol_pair_id,
                        "source_pair_id": pair.pair_id,
                        "difficulty_rank": pair.difficulty_rank,
                        "abs_z_diff": pair.abs_z_diff,
                        "snippet_i": pair.snippet_i,
                        "snippet_j": pair.snippet_j,
                        "snippet_first": first_id,
                        "snippet_second": second_id,
                        "order": order,
                        "gold_side": gold_side,
                        **info,
                        "seed": 42,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    handle.flush()
                    written += 1
                    done.add(key)

                    if written % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = written / elapsed
                        remain = (total_calls - len(done)) / rate if rate > 0 else 0
                        print(f"[GEMMA-RUN] Progress: {len(done)}/{total_calls} ({len(done)/total_calls*100:.1f}%) | Speed: {rate:.2f} calls/s | ETA: {remain/60:.1f} min | Mismatches: {mismatch_count}", flush=True)

    manifest_data = {
        "model": "google/gemma-3-12b-it",
        "model_label": "Gemma4-12B",
        "model_revision": GEMMA_REVISION,
        "device": device,
        "total_calls": total_calls,
        "completed_calls": len(done),
        "mismatch_count": mismatch_count,
        "agreement_pass_rate": (total_calls - mismatch_count) / total_calls if total_calls > 0 else 0.0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(done) == total_calls else "incomplete",
    }
    out_manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    print(f"[GEMMA-RUN] Gemma run complete. Manifest saved to {out_manifest}")

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()


def run_qwen_anchor(pairs: pd.DataFrame, lookup: dict[tuple[str, str], str], device: int = 0) -> None:
    print(f"\n=======================================================")
    print(f"[ANCHOR-RUN] Starting Qwen2.5-VL-7B Baseline Anchor Re-run on cuda:{device}...")
    print(f"=======================================================")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    out_raw = OUT_DIR / "anchor_qwen/raw_calls/qwen25_vl_7b_baseline_anchor_raw.jsonl"
    out_manifest = OUT_DIR / "manifests/qwen25_vl_7b_anchor_manifest.json"
    done = completed_keys(out_raw)
    baseline_cond = "monokai_dark__fs20__wrap80__lnon"
    total_calls = len(pairs) * 2  # 3000 * 2 = 6,000

    print(f"[ANCHOR-RUN] Planned calls: {total_calls} | Already done: {len(done)}")
    if len(done) >= total_calls:
        print("[ANCHOR-RUN] All Anchor calls already completed!")
        return

    processor = AutoProcessor.from_pretrained(QWEN_SNAPSHOT, local_files_only=True)
    tokenizer = processor.tokenizer
    cand_a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    cand_b_id = tokenizer.encode(" B", add_special_tokens=False)[0]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_SNAPSHOT,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    print("[ANCHOR-RUN] Qwen model loaded.")

    images_cache = {}
    def get_image(rel_path: str) -> Image.Image:
        if rel_path not in images_cache:
            if len(images_cache) > 200:
                images_cache.clear()
            with Image.open(ROOT / rel_path) as img:
                images_cache[rel_path] = img.convert("RGB").copy()
        return images_cache[rel_path]

    written = 0
    start_time = time.time()
    mismatch_count = 0

    with out_raw.open("a", encoding="utf-8") as handle:
        for pair in pairs.itertuples(index=False):
            prompt = PROMPT_TEMPLATE.format(language=DISPLAY_LANG[pair.language])
            for order, first_id, second_id in (("AB", pair.snippet_i, pair.snippet_j), ("BA", pair.snippet_j, pair.snippet_i)):
                key = (baseline_cond, pair.protocol_pair_id, order)
                if key in done:
                    continue

                img_first = get_image(lookup[(baseline_cond, first_id)])
                img_second = get_image(lookup[(baseline_cond, second_id)])

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "Code A image:"},
                        {"type": "image", "image": img_first},
                        {"type": "text", "text": "Code B image:"},
                        {"type": "image", "image": img_second},
                    ]
                }]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[img_first, img_second], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=24,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                input_length = inputs["input_ids"].shape[1]
                token_ids = output.sequences[0, input_length:].tolist()
                raw_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

                info = extract_logits(tokenizer, token_ids, list(output.scores), cand_a_id, cand_b_id)
                info.update({"raw_output": raw_text, "gen_tokens": len(token_ids), "capture_method": "generate_output_scores"})

                if not info["argmax_matches_parsed"]:
                    mismatch_count += 1

                gold_side = "first" if pair.human_preference == first_id else "second"
                row = {
                    "experiment": "grid",
                    "condition": baseline_cond,
                    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "model_revision": QWEN_REVISION,
                    "language": pair.language,
                    "pair_id": pair.protocol_pair_id,
                    "source_pair_id": pair.pair_id,
                    "difficulty_rank": pair.difficulty_rank,
                    "abs_z_diff": pair.abs_z_diff,
                    "snippet_i": pair.snippet_i,
                    "snippet_j": pair.snippet_j,
                    "snippet_first": first_id,
                    "snippet_second": second_id,
                    "order": order,
                    "gold_side": gold_side,
                    **info,
                    "seed": 42,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                written += 1
                done.add(key)

                if written % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = written / elapsed
                    remain = (total_calls - len(done)) / rate if rate > 0 else 0
                    print(f"[ANCHOR-RUN] Progress: {len(done)}/{total_calls} ({len(done)/total_calls*100:.1f}%) | Speed: {rate:.2f} calls/s | ETA: {remain/60:.1f} min | Mismatches: {mismatch_count}", flush=True)

    manifest_data = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": QWEN_REVISION,
        "device": device,
        "total_calls": total_calls,
        "completed_calls": len(done),
        "mismatch_count": mismatch_count,
        "agreement_pass_rate": (total_calls - mismatch_count) / total_calls if total_calls > 0 else 0.0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(done) == total_calls else "incomplete",
    }
    out_manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    print(f"[ANCHOR-RUN] Anchor run complete. Manifest saved to {out_manifest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    pairs = pd.read_csv(BASE_EXP / "data/pairs_seed42_grounded.csv")
    lookup = load_image_lookup()

    # 1. Run Gemma on 7 conditions
    run_gemma(pairs, lookup, device=args.device)

    # 2. Run Qwen Anchor on Baseline
    run_qwen_anchor(pairs, lookup, device=args.device)

    print("\n[ALL COMPLETE] Gemma-3-12b-it and Qwen Anchor re-run finished successfully!")


if __name__ == "__main__":
    main()

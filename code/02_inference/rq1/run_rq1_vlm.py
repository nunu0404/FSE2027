#!/usr/bin/env python3
"""Run the preregistered RQ1 five-model image-only battery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/rq1_model_battery_3lang_20260723"
LEGACY_PILOT = ROOT / "results/protocol_unified_3lang_20260720/code/run_logit_pilot.py"
LEGACY_RUNNER = ROOT / "experiments/rq0_viability/scripts/run_judge_pairs.py"
VERDICT_RE = re.compile(r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B)\b", re.IGNORECASE)
_LOGIT_MODULE = None

MODELS = {
    "qwen": {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "adapter": "legacy_qwen",
    },
    "internvl": {
        "id": "OpenGVLab/InternVL3-8B",
        "revision": "853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
        "adapter": "legacy_internvl",
    },
    "phi": {
        "id": "microsoft/Phi-4-multimodal-instruct",
        "revision": "93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        "adapter": "phi4mm",
        "attention_backend": "sdpa",
    },
    "gemma": {
        "id": "google/gemma-3-12b-it",
        "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "adapter": "generic_hf",
    },
    "ministral": {
        "id": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "revision": "f6fae9795746f63c9be8344932f01275f3c63734",
        "adapter": "generic_hf",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_path(model_id: str, revision: str) -> Path:
    org, name = model_id.split("/", 1)
    candidates = [
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        / f"models--{org}--{name}" / "snapshots" / revision,
        Path.home() / ".cache/huggingface/hub"
        / f"models--{org}--{name}" / "snapshots" / revision,
        Path("/ANON/scratch_rq1/hf/hub")
        / f"models--{org}--{name}" / "snapshots" / revision,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"pinned snapshot absent: {model_id}@{revision}")


def open_verified_image(relative_path: str, expected_hash: str) -> Image.Image:
    path = ROOT / relative_path
    actual = sha256(path)
    if actual != expected_hash:
        raise RuntimeError(f"image hash drift: {relative_path}: {actual} != {expected_hash}")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_pairs(phase: str) -> pd.DataFrame:
    filename = "ga0_pairs_50_seed42.csv" if phase == "ga0" else "rq1_pairs_9000.csv"
    pairs = pd.read_csv(OUT / "data" / filename)
    expected = 50 if phase == "ga0" else 9000
    if len(pairs) != expected or pairs.protocol_pair_id.nunique() != expected:
        raise ValueError(f"{phase} pair allocation drift")
    if phase == "full" and pairs.groupby("language").size().to_dict() != {
        "cuda": 3000, "java": 3000, "python": 3000
    }:
        raise ValueError("full language allocation drift")
    return pairs


def completed_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["pair_id"], row["order"])
            if key in keys:
                raise ValueError(f"duplicate output key at line {line_number}: {key}")
            keys.add(key)
    return keys


def extract_score_logits(tokenizer, token_ids: list[int], scores: list[Any]) -> dict[str, Any]:
    global _LOGIT_MODULE
    if _LOGIT_MODULE is None:
        _LOGIT_MODULE = load_module("rq1_logit_extract", LEGACY_PILOT)
    result = _LOGIT_MODULE.extract_logits(tokenizer, token_ids, scores)
    result["capture_method"] = "generate_output_scores"
    return result


class GenericHFAdapter:
    def __init__(
        self, snapshot: Path, max_new_tokens: int, fix_mistral_regex: bool = False
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        processor_kwargs = {"local_files_only": True}
        if fix_mistral_regex:
            processor_kwargs["fix_mistral_regex"] = True
        self.processor = AutoProcessor.from_pretrained(snapshot, **processor_kwargs)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        ).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, image_a: Image.Image, image_b: Image.Image):
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "Code A image:"},
            {"type": "image", "image": image_a},
            {"type": "text", "text": "Code B image:"},
            {"type": "image", "image": image_b},
        ]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_length = inputs["input_ids"].shape[1]
        token_ids = output.sequences[0, input_length:].tolist()
        raw = self.processor.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        info = extract_score_logits(self.tokenizer, token_ids, list(output.scores))
        info.update({"raw_output": raw, "gen_tokens": len(token_ids)})
        return raw, info


class PhiAdapter:
    def __init__(self, snapshot: Path, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            _attn_implementation="sdpa",
        ).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, image_a: Image.Image, image_b: Image.Image):
        formatted = (
            "<|user|><|image_1|><|image_2|>"
            + prompt
            + "\n\nThe first image is Code A. The second image is Code B."
            + "<|end|><|assistant|>"
        )
        inputs = self.processor(
            text=formatted, images=[image_a, image_b], return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_length = inputs["input_ids"].shape[1]
        token_ids = output.sequences[0, input_length:].tolist()
        raw = self.processor.batch_decode(
            [token_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        info = extract_score_logits(self.tokenizer, token_ids, list(output.scores))
        info.update({"raw_output": raw, "gen_tokens": len(token_ids)})
        return raw, info


def load_adapter(model_key: str, snapshot: Path, max_new_tokens: int):
    config = MODELS[model_key]
    if config["adapter"] in {"legacy_qwen", "legacy_internvl"}:
        runner = load_module("rq1_legacy_runner", LEGACY_RUNNER)
        pilot = load_module("rq1_legacy_pilot", LEGACY_PILOT)
        runner.set_global_seed(42)
        if config["adapter"] == "legacy_qwen":
            judge = runner.DirectQwenVLJudge(
                str(snapshot), "cuda:0", "bf16", max_new_tokens, 0.0, 1.0
            )
            return lambda prompt, a, b: pilot.qwen_generate(judge, prompt, a, b)
        judge = runner.DirectInternVLJudge(
            config["id"], "bf16", max_new_tokens, 0.0, 1.0
        )
        return lambda prompt, a, b: pilot.internvl_generate(judge, prompt, a, b)
    if config["adapter"] == "phi4mm":
        adapter = PhiAdapter(snapshot, max_new_tokens)
    else:
        adapter = GenericHFAdapter(
            snapshot,
            max_new_tokens,
            fix_mistral_regex=model_key == "ministral",
        )
    return adapter.generate


def ga0_summary(raw_path: Path, model_id: str) -> dict[str, Any]:
    frame = pd.read_json(raw_path, lines=True)
    parsed = frame.parsed_choice.notna()
    exact_output = frame.raw_output.astype(str).str.fullmatch(r"FINAL_VERDICT: [AB]")
    agreement = frame.argmax_matches_parsed.fillna(False).astype(bool)
    summary = {
        "model": model_id,
        "calls": len(frame),
        "pairs": int(frame.pair_id.nunique()),
        "parse_successes": int(parsed.sum()),
        "parse_failures": int((~parsed).sum()),
        "exact_one_line_outputs": int(exact_output.sum()),
        "non_exact_outputs": int((~exact_output).sum()),
        "argmax_matches": int(agreement.sum()),
        "argmax_mismatches": int((~agreement).sum()),
        "truncations": int((frame.gen_tokens >= 24).sum()),
        "mapping_errors": int((~frame.image_mapping_verified.astype(bool)).sum()),
        "token_families": frame.token_family.value_counts(dropna=False).to_dict(),
    }
    summary["gate_pass"] = bool(
        summary["calls"] == 100
        and summary["pairs"] == 50
        and summary["parse_failures"] == 0
        and summary["non_exact_outputs"] == 0
        and summary["argmax_mismatches"] == 0
        and summary["truncations"] == 0
        and summary["mapping_errors"] == 0
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--phase", required=True, choices=["adapter", "ga0", "full"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-calls", type=int, default=0)
    args = parser.parse_args()

    approval = json.loads((OUT / "APPROVAL_RECORD.json").read_text())
    if not approval.get("execution_authorized"):
        raise RuntimeError("author approval gate is closed")
    protocol = json.loads((OUT / "config/protocol_preregistered.json").read_text())
    frozen = protocol["inference"]
    expected = {
        "dtype": "bfloat16",
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_p_active": False,
        "max_new_tokens": 24,
        "seed": 42,
        "strict_swap": True,
        "capture_verdict_logits": True,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"protocol drift: {key}={frozen.get(key)!r}")

    config = MODELS[args.model]
    snapshot = snapshot_path(config["id"], config["revision"])
    pairs = load_pairs("ga0" if args.phase in {"adapter", "ga0"} else "full")
    if args.phase == "adapter":
        pairs = pairs.iloc[:1]
    prompt_path = (OUT / frozen["prompt"]).resolve()
    prompt_template = prompt_path.read_text().rstrip("\n")
    calls_expected = 2 if args.phase == "adapter" else len(pairs) * 2
    calls_planned = args.limit_calls or calls_expected
    if not 0 < calls_planned <= calls_expected:
        raise ValueError("invalid call limit")

    run_dir = OUT / "inference" / args.phase / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    manifest_path = run_dir / "manifest.json"
    done = completed_keys(raw_path) if args.resume else set()
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume")

    manifest = {
        "status": "running",
        "protocol_id": protocol["protocol_id"],
        "approval_commit": "2c6a439",
        "model": config["id"],
        "model_revision": config["revision"],
        "snapshot": str(snapshot),
        "adapter": config["adapter"],
        "attention_backend": config.get("attention_backend", "model_native"),
        "phase": args.phase,
        "pairs": len(pairs),
        "orders": ["AB", "BA"],
        "calls_expected": calls_expected,
        "calls_planned": calls_planned,
        "prompt_path": str(prompt_path.relative_to(ROOT)),
        "prompt_sha256": sha256(prompt_path),
        "runner_sha256": sha256(Path(__file__)),
        "environment_audit_sha256": sha256(OUT / "audit/EXECUTION_ENVIRONMENT.json"),
        "modality": "image_only",
        "packaging": "two_separately_labeled_images",
        "dtype": "bfloat16",
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p_inactive": 1.0,
                     "max_new_tokens": 24},
        "seed": 42,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": os.environ.get("RQ1_GPU_NAME"),
        "concurrency_group": os.environ.get("RQ1_CONCURRENCY_GROUP", "sequential"),
        "python": platform.python_version(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    generate = load_adapter(args.model, snapshot, 24)

    written = 0
    with raw_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for pair in pairs.itertuples(index=False):
            prompt = prompt_template.format(language=pair.language)
            orders = [
                ("AB", pair.snippet_i, pair.snippet_j, pair.image_i_path,
                 pair.image_j_path, pair.image_i_sha256, pair.image_j_sha256),
                ("BA", pair.snippet_j, pair.snippet_i, pair.image_j_path,
                 pair.image_i_path, pair.image_j_sha256, pair.image_i_sha256),
            ]
            for order, first_id, second_id, first_path, second_path, first_hash, second_hash in orders:
                key = (pair.protocol_pair_id, order)
                if key in done:
                    continue
                if len(done) + written >= calls_planned:
                    break
                image_a = open_verified_image(first_path, first_hash)
                image_b = open_verified_image(second_path, second_hash)
                _, info = generate(prompt, image_a, image_b)
                mapping_verified = (
                    (order == "AB" and first_id == pair.snippet_i and second_id == pair.snippet_j)
                    or (order == "BA" and first_id == pair.snippet_j and second_id == pair.snippet_i)
                )
                row = {
                    "phase": args.phase,
                    "model": config["id"],
                    "model_revision": config["revision"],
                    "language": pair.language,
                    "difficulty": pair.difficulty,
                    "pair_id": pair.protocol_pair_id,
                    "source_pair_id": pair.pair_id,
                    "abs_z_diff": pair.abs_z_diff,
                    "snippet_i": pair.snippet_i,
                    "snippet_j": pair.snippet_j,
                    "snippet_first": first_id,
                    "snippet_second": second_id,
                    "image_first_path": first_path,
                    "image_second_path": second_path,
                    "image_first_sha256": first_hash,
                    "image_second_sha256": second_hash,
                    "image_mapping_verified": mapping_verified,
                    "order": order,
                    "gold_side": "first" if pair.human_preference == first_id else "second",
                    **info,
                    "seed": 42,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                written += 1
                if written % 10 == 0 or args.phase == "adapter":
                    print(
                        f"{args.model} {args.phase} {len(done)+written}/{calls_planned} "
                        f"parsed={info.get('parsed_choice')} "
                        f"argmax={info.get('logit_argmax_choice')}",
                        flush=True,
                    )
            if len(done) + written >= calls_planned:
                break

    completed = len(done) + written
    manifest["status"] = "complete" if completed == calls_expected else "partial"
    manifest["calls_completed"] = completed
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.phase == "ga0" and completed == 100:
        summary = ga0_summary(raw_path, config["id"])
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        manifest["ga0_gate_pass"] = summary["gate_pass"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.phase == "ga0" and completed == 100:
        return 0 if manifest["ga0_gate_pass"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

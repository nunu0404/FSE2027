#!/usr/bin/env python3
"""Run smoke or full primary inference for the latest-generation VLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
PAIR_PATH = ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
PROMPT_PATH = OUT / "config/FROZEN_PROMPT_B.txt"
HF_CACHE = Path("/ANON/scratch_rq1/hf")
KST = timezone(timedelta(hours=9))
VERDICT_RE = re.compile(r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B)\b", re.IGNORECASE)

MODELS = {
    "qwen3": {
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "loader": "qwen3",
    },
    "internvl3_5": {
        "id": "OpenGVLab/InternVL3_5-8B-HF",
        "revision": "741a7d03020411e666c6109218ab71e08151ef86",
        "loader": "image_text_to_text",
    },
    "gemma4": {
        "id": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "loader": "multimodal_lm",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(OUT), *args], text=True
    ).strip()


def package_commit() -> str:
    return git_value("rev-parse", "HEAD")


def snapshot_path(model_id: str, revision: str) -> Path:
    org, name = model_id.split("/", 1)
    relative = Path(f"models--{org}--{name}") / "snapshots" / revision
    for base in (HF_CACHE, HF_CACHE / "hub", Path.home() / ".cache/huggingface/hub"):
        candidate = base / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"pinned model snapshot is absent: {model_id}@{revision}")


def physical_gpu() -> dict[str, str]:
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected not in {"0", "1"}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly one physical GPU: 0 or 1"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    parsed = {}
    for row in rows:
        index, uuid, name, memory, driver = row.split(", ", 4)
        parsed[index] = {
            "physical_index": index,
            "uuid": uuid,
            "name": name,
            "memory_mib": memory,
            "driver": driver,
        }
    return parsed[selected]


def fixed_smoke_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    sample = (
        pairs.sort_values(["language", "difficulty", "protocol_pair_id"])
        .groupby(["language", "difficulty"], sort=True, group_keys=False)
        .head(2)
        .copy()
    )
    if len(sample) != 18:
        raise RuntimeError("smoke set must contain 18 pairs")
    return sample


def load_pairs(phase: str) -> pd.DataFrame:
    pairs = pd.read_csv(PAIR_PATH)
    if len(pairs) != 9000 or pairs.protocol_pair_id.nunique() != 9000:
        raise RuntimeError("primary pair manifest drift")
    expected = {
        (language, difficulty): 1000
        for language in ("java", "python", "cuda")
        for difficulty in ("easy", "medium", "hard")
    }
    if pairs.groupby(["language", "difficulty"]).size().to_dict() != expected:
        raise RuntimeError("language/difficulty allocation drift")
    return fixed_smoke_pairs(pairs) if phase == "smoke" else pairs


def open_verified_image(relative_path: str, expected_sha256: str) -> Image.Image:
    path = ROOT / relative_path
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"image hash drift: {relative_path}")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def token_family(tokenizer, emitted_token_id: int) -> tuple[int, int, str]:
    candidates = {
        "space_prefixed": (
            tokenizer.encode(" A", add_special_tokens=False),
            tokenizer.encode(" B", add_special_tokens=False),
        ),
        "unprefixed": (
            tokenizer.encode("A", add_special_tokens=False),
            tokenizer.encode("B", add_special_tokens=False),
        ),
    }
    for family, (a_ids, b_ids) in candidates.items():
        if len(a_ids) == len(b_ids) == 1 and emitted_token_id in {a_ids[0], b_ids[0]}:
            return int(a_ids[0]), int(b_ids[0]), family
    raise ValueError(
        f"verdict token {emitted_token_id} is not a one-token A/B candidate: {candidates}"
    )


def candidate_token_ids(tokenizer) -> list[int]:
    values = []
    for text in (" A", " B", "A", "B"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"A/B candidate is not one token: {text!r} -> {ids}")
        values.append(int(ids[0]))
    return list(dict.fromkeys(values))


def capture_lm_head_scores(model, tokenizer):
    ids = candidate_token_ids(tokenizer)
    captured: list[dict[int, float]] = []
    language_model = getattr(model, "language_model", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and language_model is not None:
        lm_head = getattr(language_model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("could not locate LM head for read-only logit capture")

    def hook(_module, _inputs, output):
        logits = output[0] if isinstance(output, tuple) else output
        values = logits[0, -1, ids].detach().float().cpu().tolist()
        captured.append({token_id: float(value) for token_id, value in zip(ids, values)})

    return lm_head.register_forward_hook(hook), captured


def empty_logit_result() -> dict[str, Any]:
    return {
        "parsed_choice": None,
        "verdict_token_index": None,
        "verdict_token_id": None,
        "token_family": None,
        "token_id_A": None,
        "token_id_B": None,
        "logit_A": None,
        "logit_B": None,
        "margin": None,
        "logit_tie": None,
        "logit_argmax_choice": None,
        "argmax_matches_parsed": False,
        "capture_method": "lm_head_forward_hook",
    }


def extract_hook_logits(tokenizer, token_ids: list[int], captured) -> dict[str, Any]:
    if len(captured) < len(token_ids):
        raise ValueError(f"LM-head hook/token mismatch: {len(captured)} < {len(token_ids)}")
    verdict_index = None
    parsed = None
    for index in range(len(token_ids)):
        decoded = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=True)
        matches = list(VERDICT_RE.finditer(decoded))
        if matches:
            verdict_index = index
            parsed = matches[-1].group(1).upper()
            break
    if verdict_index is None or parsed is None:
        return empty_logit_result()
    emitted = int(token_ids[verdict_index])
    token_a, token_b, family = token_family(tokenizer, emitted)
    score = captured[verdict_index]
    logit_a = float(score[token_a])
    logit_b = float(score[token_b])
    argmax = "A" if logit_a >= logit_b else "B"
    return {
        "parsed_choice": parsed,
        "verdict_token_index": verdict_index,
        "verdict_token_id": emitted,
        "token_family": family,
        "token_id_A": token_a,
        "token_id_B": token_b,
        "logit_A": logit_a,
        "logit_B": logit_b,
        "margin": logit_a - logit_b,
        "logit_tie": logit_a == logit_b,
        "logit_argmax_choice": argmax,
        "argmax_matches_parsed": argmax == parsed,
        "capture_method": "lm_head_forward_hook",
    }


class ModelAdapter:
    def __init__(self, model_key: str, snapshot: Path):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.model_key = model_key
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        config = MODELS[model_key]
        common = {
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        }
        if config["loader"] == "qwen3":
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                snapshot, **common
            )
        elif config["loader"] == "image_text_to_text":
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                snapshot, trust_remote_code=True, low_cpu_mem_usage=True, **common
            )
        else:
            from transformers import AutoModelForMultimodalLM

            self.model = AutoModelForMultimodalLM.from_pretrained(snapshot, **common)
        self.model.eval()

    def generate(self, prompt: str, image_a: Image.Image, image_b: Image.Image):
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "Code A image:"},
            {"type": "image", "image": image_a},
            {"type": "text", "text": "Code B image:"},
            {"type": "image", "image": image_b},
        ]
        messages = [{"role": "user", "content": content}]
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.model_key == "gemma4":
            template_kwargs["enable_thinking"] = False
        inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        device = getattr(self.model, "device", next(self.model.parameters()).device)
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        hook, captured = capture_lm_head_scores(self.model, self.tokenizer)
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
        finally:
            hook.remove()
        input_length = inputs["input_ids"].shape[-1]
        token_ids = output.sequences[0, input_length:].tolist()
        raw = self.processor.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        info = extract_hook_logits(self.tokenizer, token_ids, captured)
        info.update(
            {
                "raw_output": raw,
                "gen_tokens": len(token_ids),
                "reached_max_new_tokens": len(token_ids) >= 24,
            }
        )
        return info


def completed_keys(path: Path) -> set[tuple[str, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["pair_id"], row["order"])
            if key in keys:
                raise RuntimeError(f"duplicate key at {path}:{line_number}: {key}")
            keys.add(key)
    return keys


def environment_snapshot(model_key: str, snapshot: Path, gpu: dict) -> dict:
    import accelerate
    import torch
    import transformers

    return {
        "python_executable": os.path.realpath(os.sys.executable),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "visible_cuda_name": torch.cuda.get_device_name(0),
        "model_key": model_key,
        "model_snapshot": str(snapshot),
    }


def write_manifest(path: Path, manifest: dict, status: str, calls: int) -> None:
    manifest["status"] = status
    manifest["calls_completed"] = calls
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def smoke_gate(raw_path: Path, repeat_path: Path, expected_calls: int) -> dict:
    frame = pd.read_json(raw_path, lines=True)
    repeats = pd.read_json(repeat_path, lines=True)
    exact = frame.raw_output.astype(str).str.fullmatch(r"FINAL_VERDICT: [AB]")
    repeat_verdicts = repeats.parsed_choice.dropna().unique().tolist()
    summary = {
        "calls": len(frame),
        "expected_calls": expected_calls,
        "pairs": frame.pair_id.nunique(),
        "duplicate_calls": int(frame[["pair_id", "order"]].duplicated().sum()),
        "parse_failures": int(frame.parsed_choice.isna().sum()),
        "missing_logits": int((frame.logit_A.isna() | frame.logit_B.isna()).sum()),
        "argmax_mismatches": int((~frame.argmax_matches_parsed.astype(bool)).sum()),
        "mapping_errors": int((~frame.image_mapping_verified.astype(bool)).sum()),
        "truncations": int(frame.reached_max_new_tokens.astype(bool).sum()),
        "non_exact_outputs": int((~exact).sum()),
        "determinism_repeat_calls": len(repeats),
        "determinism_unique_verdicts": repeat_verdicts,
    }
    summary["gate_pass"] = bool(
        summary["calls"] == expected_calls
        and summary["pairs"] == 18
        and all(
            summary[key] == 0
            for key in (
                "duplicate_calls",
                "parse_failures",
                "missing_logits",
                "argmax_mismatches",
                "mapping_errors",
                "truncations",
                "non_exact_outputs",
            )
        )
        and len(repeat_verdicts) == 1
        and len(repeats) == 3
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--phase", required=True, choices=["smoke", "full"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gpu = physical_gpu()
    config = MODELS[args.model]
    snapshot = snapshot_path(config["id"], config["revision"])
    pairs = load_pairs(args.phase)
    expected_calls = len(pairs) * 2
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    run_dir = OUT / "inference" / args.phase / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    repeat_path = run_dir / "determinism_repeats.jsonl"
    manifest_path = run_dir / "manifest.json"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume")
    done = completed_keys(raw_path) if args.resume else set()
    if args.phase == "full":
        smoke_summary = OUT / "inference/smoke" / args.model / "summary.json"
        if not smoke_summary.is_file() or not json.loads(smoke_summary.read_text()).get(
            "gate_pass"
        ):
            raise RuntimeError(f"smoke gate is not passed: {smoke_summary}")

    environment = environment_snapshot(args.model, snapshot, gpu)
    env_path = OUT / "environments" / f"{args.model}_{args.phase}.json"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_family": "latest_vlm_extension_20260830",
        "phase": args.phase,
        "model_key": args.model,
        "model_id": config["id"],
        "model_revision": config["revision"],
        "processor_revision": config["revision"],
        "model_snapshot": str(snapshot),
        "package_git_commit": package_commit(),
        "package_git_dirty_at_start": bool(git_value("status", "--porcelain")),
        "pair_manifest": str(PAIR_PATH.relative_to(ROOT)),
        "pair_manifest_sha256": sha256(PAIR_PATH),
        "prompt_path": str(PROMPT_PATH.relative_to(ROOT)),
        "prompt_sha256": sha256(PROMPT_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "parser": "FINAL[_space-]*VERDICT[:=](A|B), last match",
        "logit_orientation": "margin=logit_A-logit_B; A/B refer to displayed first/second candidate",
        "modality": "image_only",
        "packaging": "two_separate_ordered_pngs",
        "orders": ["AB", "BA"],
        "pairs": len(pairs),
        "calls_expected": expected_calls,
        "generation": {
            "dtype": "bfloat16",
            "quantization": "none",
            "batch_size": 1,
            "do_sample": False,
            "temperature_argument": "omitted",
            "temperature_effective": "not_applicable_under_greedy_decoding",
            "top_p_argument": "omitted",
            "top_p_active": False,
            "max_new_tokens": 24,
            "seed": 42,
            "seed_active": False,
            "thinking": False,
            "actual_generate_kwargs": {
                "max_new_tokens": 24,
                "do_sample": False,
                "return_dict_in_generate": True,
            },
        },
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment_path": str(env_path.relative_to(OUT)),
        "environment_sha256": sha256(env_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(manifest_path, manifest, "loading_model", len(done))
    adapter = ModelAdapter(args.model, snapshot)
    write_manifest(manifest_path, manifest, "running", len(done))

    start = time.monotonic()
    initial_done = len(done)
    with raw_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for pair in pairs.itertuples(index=False):
            prompt = prompt_template.format(language=pair.language)
            orders = (
                (
                    "AB",
                    pair.snippet_i,
                    pair.snippet_j,
                    pair.image_i_path,
                    pair.image_j_path,
                    pair.image_i_sha256,
                    pair.image_j_sha256,
                ),
                (
                    "BA",
                    pair.snippet_j,
                    pair.snippet_i,
                    pair.image_j_path,
                    pair.image_i_path,
                    pair.image_j_sha256,
                    pair.image_i_sha256,
                ),
            )
            for order, first, second, image_a, image_b, hash_a, hash_b in orders:
                key = (pair.protocol_pair_id, order)
                if key in done:
                    continue
                info = adapter.generate(
                    prompt,
                    open_verified_image(image_a, hash_a),
                    open_verified_image(image_b, hash_b),
                )
                mapping_verified = bool(
                    (order == "AB" and first == pair.snippet_i and second == pair.snippet_j)
                    or (order == "BA" and first == pair.snippet_j and second == pair.snippet_i)
                )
                row = {
                    "run_family": "latest_vlm_extension_20260830",
                    "phase": args.phase,
                    "model": config["id"],
                    "model_revision": config["revision"],
                    "language": pair.language,
                    "difficulty": pair.difficulty,
                    "pair_id": pair.protocol_pair_id,
                    "source_pair_id": pair.pair_id,
                    "abs_z_diff": pair.abs_z_diff,
                    "human_score_i_z": pair.human_score_i_z,
                    "human_score_j_z": pair.human_score_j_z,
                    "human_preference": pair.human_preference,
                    "snippet_i": pair.snippet_i,
                    "snippet_j": pair.snippet_j,
                    "snippet_first": first,
                    "snippet_second": second,
                    "image_first_path": image_a,
                    "image_second_path": image_b,
                    "image_first_sha256": hash_a,
                    "image_second_sha256": hash_b,
                    "image_mapping_verified": mapping_verified,
                    "order": order,
                    "gold_side": "first" if pair.human_preference == first else "second",
                    **info,
                    "seed": 42,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                done.add(key)
                if len(done) % 100 == 0 or args.phase == "smoke":
                    elapsed = time.monotonic() - start
                    new_calls = len(done) - initial_done
                    rate = new_calls / elapsed if elapsed else 0.0
                    remaining = expected_calls - len(done)
                    eta = datetime.now(KST) + timedelta(seconds=remaining / rate) if rate else None
                    print(
                        f"PROGRESS model={args.model} phase={args.phase} "
                        f"calls={len(done)}/{expected_calls} rate={rate:.3f}_calls_s "
                        f"eta_kst={eta.strftime('%F %T') if eta else 'NA'} "
                        f"parsed={info.get('parsed_choice')} argmax={info.get('logit_argmax_choice')}",
                        flush=True,
                    )
                if len(done) % 500 == 0:
                    write_manifest(manifest_path, manifest, "running", len(done))

    if len(done) != expected_calls:
        write_manifest(manifest_path, manifest, "incomplete", len(done))
        raise RuntimeError(f"incomplete run: {len(done)} != {expected_calls}")

    if args.phase == "smoke":
        reference_pair = pairs.iloc[0]
        prompt = prompt_template.format(language=reference_pair.language)
        repeat_rows = []
        for repeat in range(3):
            info = adapter.generate(
                prompt,
                open_verified_image(
                    reference_pair.image_i_path, reference_pair.image_i_sha256
                ),
                open_verified_image(
                    reference_pair.image_j_path, reference_pair.image_j_sha256
                ),
            )
            repeat_rows.append(
                {
                    "repeat": repeat + 1,
                    "pair_id": reference_pair.protocol_pair_id,
                    "order": "AB",
                    **info,
                }
            )
        with repeat_path.open("w", encoding="utf-8") as handle:
            for row in repeat_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        summary = smoke_gate(raw_path, repeat_path, expected_calls)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        write_manifest(
            manifest_path,
            manifest,
            "complete" if summary["gate_pass"] else "gate_failed",
            len(done),
        )
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["gate_pass"] else 2

    write_manifest(manifest_path, manifest, "complete_pending_validation", len(done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

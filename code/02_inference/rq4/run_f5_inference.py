#!/usr/bin/env python3
"""Run the preregistered 54,000-call F5 deployment-v2 inference."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/deploy_v2_logit_20260731"
RUNNER_PATH = ROOT / "experiments/rq0_viability/scripts/run_judge_pairs.py"
LOGIT_PATH = ROOT / "results/protocol_unified_3lang_20260720/code/run_logit_pilot.py"
QWEN_VL = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"
REVISIONS = {
    QWEN_VL: "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    QWEN_CODER: "c03e6d358207e414f1eca0bb1891e29f1db0e242",
}
SYSTEMS = ("direct_qwen_image_only", "source_text_qwen_coder", "rapidocr_text_qwen_coder")
EXPECTED_CALLS = 18_000
KST = timezone(timedelta(hours=9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot_path(model: str) -> Path:
    org, name = model.split("/", 1)
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--{org}--{name}"
        / "snapshots"
        / REVISIONS[model]
    )


def ref_path(model: str) -> Path:
    org, name = model.split("/", 1)
    return Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "refs/main"


def verify_static_gates() -> None:
    prereg = OUT / "config/F5_PREREGISTRATION.md"
    frozen = json.loads((OUT / "data/F5_FROZEN_INPUT_AUDIT.json").read_text())
    ocrml = json.loads((OUT / "data/F5_OCRML_GATE.json").read_text())
    lock = json.loads((OUT / "config/F5_EXECUTION_LOCK.json").read_text())
    if not frozen["gate_pass"] or not ocrml["gate_pass"]:
        raise RuntimeError("pre-inference gate failed")
    if sha256(prereg) != lock["preregistration_sha256"]:
        raise RuntimeError("preregistration changed after lock")
    if sha256(OUT / "data/F5_FROZEN_INPUT_INVENTORY.csv") != lock["frozen_inventory_sha256"]:
        raise RuntimeError("frozen inventory changed after lock")
    for model, revision in REVISIONS.items():
        if not snapshot_path(model).is_dir():
            raise FileNotFoundError(snapshot_path(model))
        if ref_path(model).read_text().strip() != revision:
            raise RuntimeError(f"model ref drift: {model}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if selected != lock["cuda_visible_devices"]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES drift: {selected}")
    expected_uuid = lock["gpu_uuid"]
    rows = [line.split(", ") for line in gpu.strip().splitlines()]
    current_uuid = dict(rows)[selected]
    if current_uuid != expected_uuid:
        raise RuntimeError(f"GPU UUID drift: {current_uuid}")


def completed_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["pair_id"], row["order"])
            if key in keys:
                raise RuntimeError(f"duplicate {path}:{number}: {key}")
            keys.add(key)
    return keys


def open_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def text_generate(judge, prompt: str, logit) -> tuple[str, dict[str, Any]]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = judge.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge.tokenizer(text, return_tensors="pt")
    device = getattr(judge.model, "device", next(judge.model.parameters()).device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    hook, captured = logit.capture_lm_head_scores(judge.model, judge.tokenizer)
    try:
        with torch.inference_mode():
            output = judge.model.generate(
                **inputs,
                max_new_tokens=judge.max_new_tokens,
                do_sample=False,
                use_cache=False,
            )
    finally:
        hook.remove()
    generated = output[0, inputs["input_ids"].shape[-1] :]
    token_ids = generated.tolist()
    raw = judge.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    info = logit.extract_hook_logits(judge.tokenizer, token_ids, captured)
    info.update({"raw_output": raw, "gen_tokens": len(token_ids)})
    del inputs, output, generated
    return raw, info


def write_manifest(system: str, status: str, calls: int, started: str, extra: dict | None = None) -> None:
    model = QWEN_VL if system == SYSTEMS[0] else QWEN_CODER
    lock = json.loads((OUT / "config/F5_EXECUTION_LOCK.json").read_text())
    manifest = {
        "run_id": "deploy_v2_logit_20260731",
        "system": system,
        "status": status,
        "model": model,
        "model_revision": REVISIONS[model],
        "model_snapshot": str(snapshot_path(model)),
        "started_at_utc": started,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calls": calls,
        "expected_calls": EXPECTED_CALLS,
        "languages": ["java", "python", "cuda"],
        "pairs_per_language": 3000,
        "orders": ["AB", "BA"],
        "dtype": "bfloat16",
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p_inactive": 1.0,
            "max_new_tokens": 24,
            "use_cache": False if system != SYSTEMS[0] else "model_default",
        },
        "batch_size": 1,
        "seed": 42,
        "capture_verdict_logits": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": lock["gpu_uuid"],
        "python": platform.python_version(),
        "execution_lock_sha256": sha256(OUT / "config/F5_EXECUTION_LOCK.json"),
        "frozen_pairs_sha256": sha256(OUT / "data/F5_FROZEN_PAIRS.csv"),
        "frozen_items_sha256": sha256(OUT / "data/F5_FROZEN_ITEMS.csv"),
        "prompt_sha256": sha256(OUT / "config/F5_FROZEN_PROMPT_B.txt"),
    }
    if extra:
        manifest.update(extra)
    path = OUT / f"inference/{system}.manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_system(
    system: str,
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    runner,
    generate: Callable[[pd.Series, pd.Series, str], tuple[str, dict[str, Any]]],
) -> None:
    raw_path = OUT / f"inference/{system}.jsonl"
    done = completed_keys(raw_path)
    started = datetime.now(timezone.utc).isoformat()
    write_manifest(system, "running", len(done), started)
    start_time = time.monotonic()
    initial_done = len(done)
    ga_rows = []
    with raw_path.open("a" if raw_path.exists() else "w", encoding="utf-8") as handle:
        for pair in pairs.itertuples(index=False):
            first_item = items.loc[pair.snippet_i]
            second_item = items.loc[pair.snippet_j]
            for order, first, second in (
                ("AB", first_item, second_item),
                ("BA", second_item, first_item),
            ):
                key = (pair.pair_id, order)
                if key in done:
                    continue
                language_prompt = runner.build_prompt("B", None, None, False, pair.language)
                raw, info = generate(first, second, language_prompt)
                row = {
                    "run_id": "deploy_v2_logit_20260731",
                    "system": system,
                    "model": QWEN_VL if system == SYSTEMS[0] else QWEN_CODER,
                    "model_revision": REVISIONS[QWEN_VL if system == SYSTEMS[0] else QWEN_CODER],
                    "language": pair.language,
                    "pair_id": pair.pair_id,
                    "difficulty": pair.difficulty,
                    "abs_z_diff": pair.abs_z_diff,
                    "snippet_i": pair.snippet_i,
                    "snippet_j": pair.snippet_j,
                    "snippet_first": first.name,
                    "snippet_second": second.name,
                    "human_preference": pair.human_preference,
                    "gold_side": "first" if pair.human_preference == first.name else "second",
                    "order": order,
                    **info,
                    "seed": 42,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                done.add(key)
                if len(ga_rows) < 20:
                    ga_rows.append(row)
                    if len(ga_rows) == 20:
                        ga = pd.DataFrame(ga_rows)
                        passed = bool(
                            ga.parsed_choice.notna().all()
                            and ga.logit_A.notna().all()
                            and ga.logit_B.notna().all()
                            and ga.argmax_matches_parsed.all()
                        )
                        ga_summary = {
                            "run_id": "deploy_v2_logit_20260731",
                            "system": system,
                            "formal_calls_used": 20,
                            "extra_pilot_calls": 0,
                            "parse_failures": int(ga.parsed_choice.isna().sum()),
                            "missing_logit_calls": int(
                                (ga.logit_A.isna() | ga.logit_B.isna()).sum()
                            ),
                            "argmax_mismatches": int((~ga.argmax_matches_parsed).sum()),
                            "gate_pass": passed,
                        }
                        (OUT / f"ga0/{system}.json").write_text(
                            json.dumps(ga_summary, indent=2) + "\n", encoding="utf-8"
                        )
                        print("GA0 " + json.dumps(ga_summary), flush=True)
                        if not passed:
                            write_manifest(system, "ga0_failed", len(done), started, ga_summary)
                            raise RuntimeError(f"GA0 failed: {system}")
                if len(done) % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    new_calls = len(done) - initial_done
                    rate = new_calls / elapsed if elapsed else 0
                    remaining = EXPECTED_CALLS - len(done)
                    eta = datetime.now(KST) + timedelta(seconds=remaining / rate) if rate else None
                    print(
                        f"PROGRESS system={system} calls={len(done)}/{EXPECTED_CALLS} "
                        f"rate={rate:.2f}_calls_s eta_kst={eta.strftime('%F %T') if eta else 'NA'}",
                        flush=True,
                    )
                if len(done) % 500 == 0:
                    write_manifest(system, "running", len(done), started)
    if len(done) != EXPECTED_CALLS:
        raise RuntimeError(f"incomplete {system}: {len(done)}")
    write_manifest(system, "complete", len(done), started)


def main() -> int:
    verify_static_gates()
    runner = load_module("f5_runner", RUNNER_PATH)
    logit = load_module("f5_logit", LOGIT_PATH)
    runner.set_global_seed(42)
    pairs = pd.read_csv(OUT / "data/F5_FROZEN_PAIRS.csv")
    items = pd.read_csv(OUT / "data/F5_FROZEN_ITEMS.csv").set_index("snippet_id")
    prompt = (OUT / "config/F5_FROZEN_PROMPT_B.txt").read_text().rstrip("\n")
    if runner.PROMPTS["B"] != prompt:
        raise RuntimeError("Prompt B drift")

    direct_path = OUT / f"inference/{SYSTEMS[0]}.jsonl"
    if len(completed_keys(direct_path)) < EXPECTED_CALLS:
        judge = runner.DirectQwenVLJudge(QWEN_VL, "auto", "bf16", 24, 0.0, 1.0)

        def direct(first, second, prompt_text):
            return logit.qwen_generate(
                judge,
                prompt_text,
                open_rgb(first.image_path),
                open_rgb(second.image_path),
            )

        run_system(SYSTEMS[0], pairs, items, runner, direct)
        del judge
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    source_path = OUT / f"inference/{SYSTEMS[1]}.jsonl"
    ocr_path = OUT / f"inference/{SYSTEMS[2]}.jsonl"
    if (
        len(completed_keys(source_path)) < EXPECTED_CALLS
        or len(completed_keys(ocr_path)) < EXPECTED_CALLS
    ):
        engine = runner.load_legacy_engine(ROOT)
        judge = engine.TextJudge(QWEN_CODER, "auto", "bf16", 24, 0.0, 1.0)

        def source(first, second, prompt_text):
            full = runner.build_prompt(
                "B", first.source_text, second.source_text, True, first.language
            )
            return text_generate(judge, full, logit)

        def rapid(first, second, prompt_text):
            full = runner.build_prompt(
                "B", first.rapidocr_text, second.rapidocr_text, True, first.language
            )
            return text_generate(judge, full, logit)

        if len(completed_keys(source_path)) < EXPECTED_CALLS:
            run_system(SYSTEMS[1], pairs, items, runner, source)
        if len(completed_keys(ocr_path)) < EXPECTED_CALLS:
            run_system(SYSTEMS[2], pairs, items, runner, rapid)
        del judge
        gc.collect()

    total = sum(
        len(completed_keys(OUT / f"inference/{system}.jsonl")) for system in SYSTEMS
    )
    if total != 54_000:
        raise RuntimeError(f"F5 total call mismatch: {total}")
    print(json.dumps({"run_id": "deploy_v2_logit_20260731", "status": "complete", "calls": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

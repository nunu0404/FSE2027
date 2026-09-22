#!/usr/bin/env python3
"""Run controlled larger-model cross-family readability inference on Java pairs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

# Set Hugging Face cache directory to high-capacity storage
os.environ.setdefault("HF_HOME", "/ANON/hf_cache")
TOKEN_PATH = Path("/ANON/scratch_rq1/hf/token")
HF_TOKEN = TOKEN_PATH.read_text().strip() if TOKEN_PATH.exists() else None

EXP_ROOT = Path("/ANON/experiment_root")
RESULTS_DIR = EXP_ROOT / "results/large_model_cross_family_3model_300java_20260910"
RENDER_DIR = EXP_ROOT / "experiments/rq0_viability/data/rendered/default"
PAIR_IDS_CSV = RESULTS_DIR / "pair_ids.csv"

FROZEN_PROMPT = """You are comparing two Java code snippets by readability only.

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

MODEL_REGISTRY = {
    "qwen": {
        "model_name": "Qwen2.5-VL-32B-Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "exact_revision": "7cfb30d71a1f4f49a57592323337a4a4727301da",
        "raw_jsonl": RESULTS_DIR / "qwen25_vl_32b_raw.jsonl",
    },
    "gemma": {
        "model_name": "gemma-3-27b-it",
        "repo_id": "google/gemma-3-27b-it",
        "exact_revision": "005ad3404e59d6023443cb575daa05336842228a",
        "raw_jsonl": RESULTS_DIR / "gemma3_27b_raw.jsonl",
    },
    "mistral": {
        "model_name": "Mistral-Small-3.1-24B-Instruct-2503",
        "repo_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "exact_revision": "68faf511d618ef198fef186659617cfd2eb8e33a",
        "raw_jsonl": RESULTS_DIR / "mistral_small_31_24b_raw.jsonl",
    },
}


def parse_choice(text: str | None) -> str | None:
    """Standard frozen verdict parser conforming to Figure 5 output contract."""
    if not text:
        return None
    cleaned = text.strip()
    matches = list(re.finditer(r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B)\b", cleaned, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper()
    # Strip quotes, backticks, punctuation, and whitespace
    stripped = cleaned.strip("`\"' \t\r\n").rstrip(".!:;").strip().upper()
    if stripped in {"A", "B"}:
        return stripped
    # Strict start-of-string single char check if no other tokens
    m_start = re.match(r"^([AB])\b", cleaned, re.IGNORECASE)
    if m_start and len(cleaned.split()) == 1:
        return m_start.group(1).upper()
    return None


def map_choice(choice: str | None, order: str, snippet_x: str, snippet_y: str) -> str | None:
    """Map verdict A or B to the actual snippet ID based on presentation order.
    
    AB order: Code A = snippet_x, Code B = snippet_y
      Choice A -> snippet_x
      Choice B -> snippet_y
    BA order: Code A = snippet_y, Code B = snippet_x
      Choice A -> snippet_y
      Choice B -> snippet_x
    """
    if choice == "A":
        return snippet_x if order == "AB" else snippet_y
    elif choice == "B":
        return snippet_y if order == "AB" else snippet_x
    return None


def run_unit_tests() -> bool:
    """Unit tests for mapping, parsing, and data contracts."""
    print("[UNIT-TEST] Running unit tests...")
    
    # 1. Parser tests
    assert parse_choice("FINAL_VERDICT: A") == "A"
    assert parse_choice("FINAL_VERDICT: B") == "B"
    assert parse_choice("FINAL_VERDICT: a") == "A"
    assert parse_choice("Final Verdict: B") == "B"
    assert parse_choice("FINAL-VERDICT: A") == "A"
    assert parse_choice("A") == "A"
    assert parse_choice("B") == "B"
    assert parse_choice("A.") == "A"
    assert parse_choice("`B`") == "B"
    assert parse_choice("I think snippet A is better because...") is None
    assert parse_choice("Both snippets are readable.") is None
    assert parse_choice("") is None
    assert parse_choice(None) is None
    print("  [PASS] Parser tests passed.")

    # 2. Mapping tests
    sx, sy = "snippet_X", "snippet_Y"
    # AB order
    assert map_choice("A", "AB", sx, sy) == sx
    assert map_choice("B", "AB", sx, sy) == sy
    assert map_choice(None, "AB", sx, sy) is None
    # BA order
    assert map_choice("A", "BA", sx, sy) == sy
    assert map_choice("B", "BA", sx, sy) == sx
    assert map_choice(None, "BA", sx, sy) is None
    print("  [PASS] Choice mapping tests passed.")

    # 3. Strict-swap validity logic
    # Consistent choice of sx: AB -> A (sx), BA -> B (sx) => Valid
    ch_ab, ch_ba = "A", "B"
    mapped_ab = map_choice(ch_ab, "AB", sx, sy)
    mapped_ba = map_choice(ch_ba, "BA", sx, sy)
    assert mapped_ab == mapped_ba == sx
    
    # Consistent choice of sy: AB -> B (sy), BA -> A (sy) => Valid
    ch_ab, ch_ba = "B", "A"
    mapped_ab = map_choice(ch_ab, "AB", sx, sy)
    mapped_ba = map_choice(ch_ba, "BA", sx, sy)
    assert mapped_ab == mapped_ba == sy

    # Inconsistent choice (position bias A): AB -> A (sx), BA -> A (sy) => Swap error
    ch_ab, ch_ba = "A", "A"
    mapped_ab = map_choice(ch_ab, "AB", sx, sy)
    mapped_ba = map_choice(ch_ba, "BA", sx, sy)
    assert mapped_ab != mapped_ba
    print("  [PASS] Strict-swap logic tests passed.")

    print("[UNIT-TEST] All unit tests PASSED successfully!\n")
    return True


class ModelRunner:
    def __init__(self, model_key: str, device_id: int = 0) -> None:
        self.model_key = model_key
        self.meta = MODEL_REGISTRY[model_key]
        self.repo_id = self.meta["repo_id"]
        self.revision = self.meta["exact_revision"]
        self.device = torch.device(f"cuda:{device_id}")
        self.device_id = device_id
        
        print(f"[{self.meta['model_name']}] Initializing on GPU {device_id}...")
        self._load_processor_and_model()

    def _load_processor_and_model(self) -> None:
        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "token": HF_TOKEN,
            "torch_dtype": torch.bfloat16,
            "device_map": { "": self.device_id },
        }
        
        # Load processor
        proc_kwargs: dict[str, Any] = {"revision": self.revision, "token": HF_TOKEN}
        if self.model_key == "mistral":
            proc_kwargs["fix_mistral_regex"] = True

        self.processor = AutoProcessor.from_pretrained(self.repo_id, **proc_kwargs)
        
        # Load model using AutoModelForImageTextToText / AutoModelForVision2Seq
        print(f"[{self.meta['model_name']}] Loading weights in bfloat16...")
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(self.repo_id, **kwargs)
        except Exception:
            self.model = AutoModelForVision2Seq.from_pretrained(self.repo_id, **kwargs)
        
        self.model.eval()
        vram_used = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        print(f"[{self.meta['model_name']}] Model loaded. VRAM allocated: {vram_used:.2f} GB")

    def infer(self, img_a: Image.Image, img_b: Image.Image, prompt_text: str) -> dict[str, Any]:
        """Run single image-only inference with separate 2-image packaging."""
        system_instruction = (
            "You are comparing code readability. "
            "You must output ONLY the requested final verdict line with NO explanation."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Code A image:\n"},
                    {"type": "image"},
                    {"type": "text", "text": "\n\nCode B image:\n"},
                    {"type": "image"},
                    {"type": "text", "text": f"\n\n{prompt_text}"},
                ],
            },
        ]
        
        rendered_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Prepare inputs
        inputs = self.processor(
            text=[rendered_prompt],
            images=[[img_a, img_b]],
            return_tensors="pt",
        )
        # Ensure floating tensors match model dtype (e.g. bfloat16 for Pixtral/Mistral vision tower)
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        inputs = {
            k: (
                v.to(self.device, dtype=model_dtype)
                if hasattr(v, "is_floating_point") and v.is_floating_point()
                else v.to(self.device)
                if hasattr(v, "to")
                else v
            )
            for k, v in inputs.items()
        }
        
        start_time = time.monotonic()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
            )
        latency = time.monotonic() - start_time
        
        # Extract new tokens only
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        response_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        return {
            "raw_prompt": rendered_prompt,
            "raw_response": response_text,
            "generated_tokens": generated_ids.shape[1],
            "latency_sec": round(latency, 4),
        }

    def cleanup(self) -> None:
        """Release GPU memory cleanly."""
        print(f"[{self.meta['model_name']}] Cleaning up model from GPU {self.device_id}...")
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        vram_used = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        print(f"[{self.meta['model_name']}] Cleanup complete. Remaining VRAM: {vram_used:.2f} GB")


def load_pairs(csv_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if limit is not None and limit < len(df):
        # Balanced sampling across difficulty strata to yield exactly `limit` pairs
        diffs = ["easy", "medium", "hard"]
        selected_rows = []
        counts = {d: 0 for d in diffs}
        while len(selected_rows) < limit:
            for d in diffs:
                if len(selected_rows) >= limit:
                    break
                subset = df[df["difficulty"] == d]
                idx = counts[d]
                if idx < len(subset):
                    selected_rows.append(subset.iloc[idx])
                    counts[d] += 1
        df = pd.DataFrame(selected_rows)
    return df.to_dict(orient="records")


def load_image(snippet_id: str) -> Image.Image:
    path = RENDER_DIR / f"{snippet_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Image not found for snippet {snippet_id} at {path}")
    with Image.open(path) as img:
        return img.convert("RGB")


def run_experiment_for_model(
    model_key: str,
    pairs: list[dict[str, Any]],
    output_path: Path,
    device_id: int = 0,
    is_smoke: bool = False,
) -> pd.DataFrame:
    runner = ModelRunner(model_key, device_id=device_id)
    
    # Load completed keys for resuming
    completed_keys: set[str] = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_keys.add(f"{record['pair_id']}__{record['order']}")
                    except json.JSONDecodeError:
                        pass
        print(f"[{runner.meta['model_name']}] Found {len(completed_keys)} previously completed calls.")

    records: list[dict[str, Any]] = []
    total_calls = len(pairs) * 2
    processed_count = len(completed_keys)
    
    print(f"[{runner.meta['model_name']}] Starting inference on {len(pairs)} pairs ({total_calls} calls)...")
    
    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, pair in enumerate(pairs, 1):
            pair_id = pair["pair_id"]
            sx = pair["snippet_x_id"]
            sy = pair["snippet_y_id"]
            diff = pair["difficulty"]
            human_pref = pair["human_preference"]
            
            img_x = load_image(sx)
            img_y = load_image(sy)
            
            for order in ["AB", "BA"]:
                call_key = f"{pair_id}__{order}"
                if call_key in completed_keys:
                    continue
                
                # AB order: Code A = sx, Code B = sy
                # BA order: Code A = sy, Code B = sx
                img_a, img_b = (img_x, img_y) if order == "AB" else (img_y, img_x)
                snippet_a, snippet_b = (sx, sy) if order == "AB" else (sy, sx)
                
                call_result = runner.infer(img_a, img_b, FROZEN_PROMPT)
                raw_resp = call_result["raw_response"]
                parsed = parse_choice(raw_resp)
                choice_snippet = map_choice(parsed, order, sx, sy)
                
                record = {
                    "call_key": call_key,
                    "model_key": model_key,
                    "model_name": runner.meta["model_name"],
                    "exact_revision": runner.revision,
                    "pair_id": pair_id,
                    "order": order,
                    "snippet_x_id": sx,
                    "snippet_y_id": sy,
                    "code_a_snippet": snippet_a,
                    "code_b_snippet": snippet_b,
                    "difficulty": diff,
                    "human_preference": human_pref,
                    "raw_prompt": call_result["raw_prompt"],
                    "raw_response": raw_resp,
                    "generated_tokens": call_result["generated_tokens"],
                    "parsed_verdict": parsed,
                    "actual_choice_snippet": choice_snippet,
                    "latency_sec": call_result["latency_sec"],
                    "parse_failure": parsed is None,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                records.append(record)
                processed_count += 1
                
                if processed_count % 10 == 0 or processed_count == total_calls:
                    print(
                        f"[{runner.meta['model_name']}] Progress: {processed_count}/{total_calls} calls "
                        f"({processed_count / total_calls * 100:.1f}%) | Pair {idx}/{len(pairs)} | "
                        f"Last: {order} -> '{parsed}' ({choice_snippet})",
                        flush=True,
                    )

    runner.cleanup()
    print(f"[{runner.meta['model_name']}] Finished all calls. Saved to {output_path}.\n")
    
    # Return complete dataframe of all records for this model
    all_rows = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_rows.append(json.loads(line))
    return pd.DataFrame(all_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run larger-model cross-family reliability experiment.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen", "gemma", "mistral", "all"],
        default=["all"],
        help="Models to evaluate.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0).")
    parser.add_argument("--smoke-test", action="store_true", help="Run 10-pair smoke test.")
    parser.add_argument("--unit-test-only", action="store_true", help="Run unit tests and exit.")
    parser.add_argument("--limit-pairs", type=int, default=None, help="Limit number of pairs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Always run unit tests first
    if not run_unit_tests():
        sys.exit(1)
    if args.unit_test_only:
        print("[INFO] Unit tests completed. Exiting.")
        sys.exit(0)

    target_models = ["qwen", "gemma", "mistral"] if "all" in args.models else args.models
    
    if args.smoke_test:
        print("=" * 60)
        print("STARTING 10-PAIR SMOKE TEST ACROSS 3 MODELS")
        print("=" * 60)
        pairs = load_pairs(PAIR_IDS_CSV, limit=10)
        print(f"Loaded {len(pairs)} smoke test pairs:")
        for p in pairs:
            print(f"  {p['pair_id']} (diff: {p['difficulty']}, pref: {p['human_preference']})")
        
        smoke_dir = RESULTS_DIR / "smoke_test"
        smoke_dir.mkdir(exist_ok=True)
        
        for m_key in target_models:
            smoke_out = smoke_dir / f"{m_key}_smoke_raw.jsonl"
            df_res = run_experiment_for_model(
                m_key, pairs, smoke_out, device_id=args.gpu, is_smoke=True
            )
            # Evaluate parse failure rate
            fail_rate = df_res["parse_failure"].mean()
            print(f"[SMOKE-TEST] {m_key} parse failure rate: {fail_rate:.1%} ({df_res['parse_failure'].sum()}/{len(df_res)})")
            if fail_rate > 0.20:
                print(f"[ERROR] {m_key} parse failure rate exceeded 20%! Aborting as per protocol.", file=sys.stderr)
                sys.exit(1)
        print("\n[SMOKE-TEST] All smoke tests PASSED successfully!\n")
        return

    # Full experiment run
    print("=" * 60)
    print("STARTING FULL 300-PAIR EXPERIMENT ACROSS MODELS")
    print(f"Models: {target_models} on GPU {args.gpu}")
    print("=" * 60)
    pairs = load_pairs(PAIR_IDS_CSV, limit=args.limit_pairs)
    print(f"Loaded {len(pairs)} pairs for evaluation.")
    
    for m_key in target_models:
        meta = MODEL_REGISTRY[m_key]
        out_path = meta["raw_jsonl"]
        run_experiment_for_model(m_key, pairs, out_path, device_id=args.gpu, is_smoke=False)

    print("=" * 60)
    print("ALL MODEL RUNS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

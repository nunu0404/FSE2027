#!/usr/bin/env python3
"""Run controlled larger-model cross-family reliability experiment on Multi-Language benchmark (Java, Python, CUDA)."""

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
)

# Hugging Face storage settings
os.environ.setdefault("HF_HOME", "/ANON/hf_cache")
TOKEN_PATH = Path("/ANON/scratch_rq1/hf/token")
HF_TOKEN = TOKEN_PATH.read_text().strip() if TOKEN_PATH.exists() else None

EXP_ROOT = Path("/ANON/experiment_root")
RESULTS_DIR = EXP_ROOT / "results/multilang_cross_family_3model_python_cuda_java_20260910"
PREV_JAVA_DIR = EXP_ROOT / "results/large_model_cross_family_3model_300java_20260910"
PAIR_IDS_CSV = RESULTS_DIR / "pair_ids_multilang.csv"

# Frozen Prompt Template with {language} parameter
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

MODEL_REGISTRY = {
    "qwen": {
        "model_name": "Qwen2.5-VL-32B-Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "exact_revision": "7cfb30d71a1f4f49a57592323337a4a4727301da",
        "raw_jsonl": RESULTS_DIR / "qwen25_vl_32b_multilang_raw.jsonl",
        "prev_java_raw": PREV_JAVA_DIR / "qwen25_vl_32b_raw.jsonl",
    },
    "gemma": {
        "model_name": "gemma-3-27b-it",
        "repo_id": "google/gemma-3-27b-it",
        "exact_revision": "005ad3404e59d6023443cb575daa05336842228a",
        "raw_jsonl": RESULTS_DIR / "gemma3_27b_multilang_raw.jsonl",
        "prev_java_raw": PREV_JAVA_DIR / "gemma3_27b_raw.jsonl",
    },
    "mistral": {
        "model_name": "Mistral-Small-3.1-24B-Instruct-2503",
        "repo_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "exact_revision": "68faf511d618ef198fef186659617cfd2eb8e33a",
        "raw_jsonl": RESULTS_DIR / "mistral_small_31_24b_multilang_raw.jsonl",
        "prev_java_raw": PREV_JAVA_DIR / "mistral_small_31_24b_raw.jsonl",
    },
}

FINAL_RE = re.compile(r"FINAL[_\s-]*VERDICT\s*[:=]\s*([AB])\b", re.I)
ANSWER_RE = re.compile(r"\b(?:answer|choice)\s*[:=]\s*[\"']?([AB])\b", re.I)


def parse_verdict(raw_response: str) -> str | None:
    if not raw_response or not isinstance(raw_response, str):
        return None
    cleaned = raw_response.strip()
    if cleaned in {"A", "B", "A.", "B.", "(A)", "(B)", "**A**", "**B**"}:
        return cleaned.strip("().*").upper()
    m = FINAL_RE.search(cleaned)
    if m:
        return m.group(1).upper()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        if line in {"A", "B", "A.", "B.", "(A)", "(B)", "**A**", "**B**"}:
            return line.strip("().*").upper()
        m = FINAL_RE.search(line)
        if m:
            return m.group(1).upper()
        m = ANSWER_RE.search(line)
        if m:
            return m.group(1).upper()
    return None


def map_choice_to_snippet(order: str, verdict: str | None, snip_x: str, snip_y: str) -> str | None:
    if verdict not in {"A", "B"}:
        return None
    if order == "AB":
        return snip_x if verdict == "A" else snip_y
    elif order == "BA":
        return snip_y if verdict == "A" else snip_x
    raise ValueError(f"Invalid order: {order}")


def run_unit_tests() -> bool:
    print("[UNIT-TEST] Running unit tests for multi-language evaluation...")
    assert parse_verdict("FINAL_VERDICT: A") == "A"
    assert parse_verdict("FINAL_VERDICT: B") == "B"
    assert parse_verdict("A") == "A"
    assert parse_verdict("Analysis: blah\nFINAL_VERDICT: B") == "B"
    assert parse_verdict("I think code A is better because...") is None

    assert map_choice_to_snippet("AB", "A", "x1", "y1") == "x1"
    assert map_choice_to_snippet("AB", "B", "x1", "y1") == "y1"
    assert map_choice_to_snippet("BA", "A", "x1", "y1") == "y1"
    assert map_choice_to_snippet("BA", "B", "x1", "y1") == "x1"
    print("[UNIT-TEST] All unit tests PASSED successfully!\n")
    return True


class MultiLangRunner:
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
        
        proc_kwargs: dict[str, Any] = {"revision": self.revision, "token": HF_TOKEN}
        if self.model_key == "mistral":
            proc_kwargs["fix_mistral_regex"] = True

        self.processor = AutoProcessor.from_pretrained(self.repo_id, **proc_kwargs)
        
        print(f"[{self.meta['model_name']}] Loading weights in bfloat16...")
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(self.repo_id, **kwargs)
        except Exception:
            self.model = AutoModelForVision2Seq.from_pretrained(self.repo_id, **kwargs)
        
        self.model.eval()
        vram_used = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        print(f"[{self.meta['model_name']}] Model loaded. VRAM allocated: {vram_used:.2f} GB")

    def infer(self, img_a: Image.Image, img_b: Image.Image, prompt_text: str) -> dict[str, Any]:
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
        
        inputs = self.processor(
            text=[rendered_prompt],
            images=[[img_a, img_b]],
            return_tensors="pt",
        )
        
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
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        response_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        return {
            "raw_prompt": rendered_prompt,
            "raw_response": response_text,
            "generated_tokens": len(generated_ids[0]),
            "latency_sec": round(latency, 4),
        }

    def cleanup(self) -> None:
        print(f"[{self.meta['model_name']}] Cleaning up model from GPU {self.device_id}...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        vram_rem = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        print(f"[{self.meta['model_name']}] Cleanup complete. Remaining VRAM: {vram_rem:.2f} GB")


def evaluate_multilang(runner: MultiLangRunner, pairs_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load previously completed calls in destination file
    completed_keys: set[str] = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_keys.add(record["call_key"])
                    except Exception:
                        pass
    
    # 2. Check if previous Java raw results can be imported
    prev_java_path = runner.meta.get("prev_java_raw")
    imported_java_count = 0
    if prev_java_path and prev_java_path.exists():
        with open(prev_java_path, "r", encoding="utf-8") as f_in, open(output_path, "a", encoding="utf-8") as f_out:
            for line in f_in:
                if line.strip():
                    rec = json.loads(line)
                    k = rec["call_key"]
                    if k not in completed_keys:
                        # Append language tag if missing
                        if "language" not in rec:
                            rec["language"] = "java"
                        f_out.write(json.dumps(rec) + "\n")
                        completed_keys.add(k)
                        imported_java_count += 1
        if imported_java_count > 0:
            print(f"[{runner.meta['model_name']}] Imported {imported_java_count} existing Java calls from previous run.")
    
    total_calls = len(pairs_df) * 2
    calls_done = len(completed_keys)
    print(f"[{runner.meta['model_name']}] Total calls planned: {total_calls} (already completed: {calls_done})")
    
    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx, pair in pairs_df.iterrows():
            pair_id = pair["pair_id"]
            lang = pair["language"]
            diff = pair["difficulty"]
            snip_x = pair["snippet_x_id"]
            snip_y = pair["snippet_y_id"]
            pref = pair["human_preference"]
            img_x_path = pair["image_x_path"]
            img_y_path = pair["image_y_path"]
            
            # Format language-specific prompt
            lang_name_str = "Java" if lang == "java" else ("Python" if lang == "python" else "CUDA")
            prompt_text = PROMPT_TEMPLATE.format(language=lang_name_str)
            
            # Image objects cache per pair
            img_x = None
            img_y = None
            
            for order in ["AB", "BA"]:
                call_key = f"{pair_id}__{order}"
                if call_key in completed_keys:
                    continue
                
                if img_x is None:
                    img_x = Image.open(img_x_path).convert("RGB")
                    img_y = Image.open(img_y_path).convert("RGB")
                
                if order == "AB":
                    img_a, img_b = img_x, img_y
                    code_a_snip, code_b_snip = snip_x, snip_y
                else:
                    img_a, img_b = img_y, img_x
                    code_a_snip, code_b_snip = snip_y, snip_x
                
                try:
                    res = runner.infer(img_a, img_b, prompt_text)
                    raw_resp = res["raw_response"]
                    parsed = parse_verdict(raw_resp)
                    choice = map_choice_to_snippet(order, parsed, snip_x, snip_y)
                    parse_fail = (parsed is None)
                    err_msg = None
                except Exception as exc:
                    print(f"\n[ERROR] Call failed for {call_key}: {exc}")
                    res = {"raw_prompt": "", "raw_response": "", "generated_tokens": 0, "latency_sec": 0.0}
                    raw_resp = ""
                    parsed = None
                    choice = None
                    parse_fail = True
                    err_msg = str(exc)
                
                record = {
                    "call_key": call_key,
                    "model_key": runner.model_key,
                    "model_name": runner.meta["model_name"],
                    "exact_revision": runner.meta["exact_revision"],
                    "pair_id": pair_id,
                    "language": lang,
                    "order": order,
                    "snippet_x_id": snip_x,
                    "snippet_y_id": snip_y,
                    "code_a_snippet": code_a_snip,
                    "code_b_snippet": code_b_snip,
                    "difficulty": diff,
                    "human_preference": pref,
                    "raw_prompt": res["raw_prompt"],
                    "raw_response": raw_resp,
                    "generated_tokens": res["generated_tokens"],
                    "parsed_verdict": parsed,
                    "actual_choice_snippet": choice,
                    "latency_sec": res["latency_sec"],
                    "parse_failure": parse_fail,
                    "error": err_msg,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                completed_keys.add(call_key)
                calls_done += 1
                
                if calls_done % 20 == 0 or calls_done == total_calls:
                    pct = (calls_done / total_calls) * 100
                    print(f"[{runner.meta['model_name']}] Progress: {calls_done}/{total_calls} ({pct:.1f}%) | Last: {lang} {order} -> '{parsed}' ({choice})")

    print(f"[{runner.meta['model_name']}] Finished all calls. Output: {output_path}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["qwen", "gemma", "mistral", "all"], default=["all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--unit-test-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not run_unit_tests():
        sys.exit(1)
    if args.unit_test_only:
        sys.exit(0)

    target_models = ["qwen", "gemma", "mistral"] if "all" in args.models else args.models
    df_pairs = pd.read_csv(PAIR_IDS_CSV)
    
    if args.smoke_test:
        print("=" * 60)
        print("STARTING MULTI-LANGUAGE SMOKE TEST (10 pairs per language = 30 pairs total)")
        print("=" * 60)
        # Sample 10 pairs per language (easy, med, hard balanced)
        smoke_pairs = []
        for lang in ["java", "python", "cuda"]:
            smoke_pairs.append(df_pairs[df_pairs["language"] == lang].head(10))
        smoke_df = pd.concat(smoke_pairs).reset_index(drop=True)
        print(f"Total smoke test pairs: {len(smoke_df)}")
        
        smoke_dir = RESULTS_DIR / "smoke_test"
        smoke_dir.mkdir(exist_ok=True)
        
        for m_key in target_models:
            runner = MultiLangRunner(m_key, device_id=args.gpu)
            out_file = smoke_dir / f"{m_key}_multilang_smoke_raw.jsonl"
            evaluate_multilang(runner, smoke_df, out_file)
            runner.cleanup()
            
            # Check parse failure rate
            fails = 0
            cnt = 0
            with open(out_file, "r") as f:
                for line in f:
                    if line.strip():
                        cnt += 1
                        if json.loads(line).get("parse_failure", False):
                            fails += 1
            fail_rate = (fails / cnt) * 100 if cnt > 0 else 0.0
            print(f"[SMOKE-TEST] {m_key} parse failure rate: {fail_rate:.1f}% ({fails}/{cnt})")
            if fail_rate > 20.0:
                print(f"[ERROR] {m_key} parse failure exceeded 20%! Aborting.")
                sys.exit(1)
        print("\n[SMOKE-TEST] All multi-language smoke tests PASSED successfully!\n")
        return

    # Full Run
    print("=" * 60)
    print("STARTING FULL MULTI-LANGUAGE EVALUATION (900 PAIRS)")
    print(f"Models: {target_models} on GPU {args.gpu}")
    print("=" * 60)
    
    for m_key in target_models:
        runner = MultiLangRunner(m_key, device_id=args.gpu)
        evaluate_multilang(runner, df_pairs, runner.meta["raw_jsonl"])
        runner.cleanup()

    print("=" * 60)
    print("ALL MULTI-LANGUAGE MODEL RUNS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()

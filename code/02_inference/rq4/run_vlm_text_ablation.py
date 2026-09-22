#!/usr/bin/env python3
import os
import json
import torch
import re
import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm

OUT_DIR = Path("/ANON/experiment_root/results/text_ablation_20260920")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = (
    "You are comparing two {language} code snippets by readability only.\n\n"
    "Readability means how easily a developer can understand the local structure and intent of the code.\n\n"
    "Consider:\n"
    "1. Visual clarity: indentation consistency, spacing, line breaks, density.\n"
    "2. Structural readability: block separation and control-flow traceability.\n"
    "3. Information efficiency: naming clarity and local comprehensibility when text is legible.\n\n"
    "Ignore functional correctness unless the code is locally corrupted or uninterpretable.\n\n"
    "Code A:\n```\n{text_a}\n```\n\n"
    "Code B:\n```\n{text_b}\n```\n\n"
    "Return exactly one line:\n"
    "FINAL_VERDICT: A\n"
    "or\n"
    "FINAL_VERDICT: B"
)

VERDICT_RE = re.compile(r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B)\b", re.IGNORECASE)

def extract_logits(tokenizer, token_ids, scores):
    verdict_index = None
    parsed = None
    for index in range(len(token_ids)):
        decoded = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=True)
        match = VERDICT_RE.search(decoded)
        if match:
            verdict_index = index
            parsed = match.group(1).upper()
            break
            
    if verdict_index is None or parsed is None:
        return {"parsed_choice": None}

    verdict_token_id = token_ids[verdict_index]
    
    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    
    if verdict_token_id not in {a_id, b_id}:
        a_id = tokenizer.encode("A", add_special_tokens=False)[0]
        b_id = tokenizer.encode("B", add_special_tokens=False)[0]
        
    logits = scores[verdict_index][0]
    return {
        "parsed_choice": parsed,
        "logit_A": float(logits[a_id]),
        "logit_B": float(logits[b_id])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["ocr", "source"], required=True)
    args = parser.parse_args()
    cond = args.condition

    print("Loading OCR and Source texts...")
    java_ocr_df = pd.read_csv("/ANON/experiment_root/results/screenshot_only_ocr_clean_20260707/clean_ocr_text_by_snippet_all_engines.csv")
    pycuda_ocr_df = pd.read_csv("/ANON/experiment_root/results/python_cuda_missing_experiments_20260716/ocr/ocr_text_by_snippet_all_engines.csv")

    ocr_df = pd.concat([java_ocr_df, pycuda_ocr_df], ignore_index=True)
    rapidocr_df = ocr_df[ocr_df.ocr_engine == "rapidocr"].set_index("snippet_id")

    texts_dict = {}
    for snippet_id, row in rapidocr_df.iterrows():
        # Fallback to empty string if nan
        src = str(row["original_source_text"]) if pd.notna(row["original_source_text"]) else ""
        ocr = str(row["ocr_text"]) if pd.notna(row["ocr_text"]) else ""
        texts_dict[snippet_id] = {
            "source": src,
            "ocr": ocr
        }

    print("Loading pairs...")
    java_pairs_path = "/ANON/experiment_root/experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl"
    pycuda_pairs_path = "/ANON/experiment_root/results/python_cuda_vlm_main_20260715/outputs/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl"

    pairs = []
    for p in [java_pairs_path, pycuda_pairs_path]:
        with open(p, "r") as f:
            for line in f:
                pairs.append(json.loads(line.strip()))

    print(f"Loaded {len(pairs)} pairs.")

    print("Loading model...")
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    out_file = OUT_DIR / f"qwen2.5_vl_7b_text_only_{cond}.jsonl"
    # Resume capability
    processed_run_keys = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            for line in f:
                processed_run_keys.add(json.loads(line)["run_pair_key"])
        print(f"Resuming {cond}. {len(processed_run_keys)} already done.")

    with open(out_file, "a") as f:
        for pair in tqdm(pairs, desc=f"Running {cond}"):
            for order in ["ab", "ba"]:
                run_key = f"{pair['pair_id']}__{order}"
                if run_key in processed_run_keys:
                    continue

                if order == "ab":
                    snippet_a_id = pair["snippet_i"]
                    snippet_b_id = pair["snippet_j"]
                else:
                    snippet_a_id = pair["snippet_j"]
                    snippet_b_id = pair["snippet_i"]
                
                # FIX 2026-09-20: dataset_name_i is only Buse/Dorn/Scalabrino and never
                # contains "python"/"cuda", so the old rule labelled ALL 9,000 pairs "Java".
                # Snippet ids do carry the language (dorn_cuda_*, dorn_python_*, rq0_* = java).
                _sid = str(pair["snippet_i"]).lower()
                if "cuda" in _sid: language = "cuda"
                elif "python" in _sid: language = "python"
                else: language = "java"

                text_a = texts_dict[snippet_a_id][cond]
                text_b = texts_dict[snippet_b_id][cond]
                
                prompt = PROMPT_TEMPLATE.format(language=language.capitalize(), text_a=text_a, text_b=text_b)
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt}
                    ]}
                ]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs, 
                        max_new_tokens=24,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                input_length = inputs.input_ids.shape[1]
                gen_tokens = output.sequences[0, input_length:].tolist()
                raw_output = processor.decode(gen_tokens, skip_special_tokens=True)
                
                info = extract_logits(processor.tokenizer, gen_tokens, output.scores)
                
                result = pair.copy()
                result.update({
                    "run_pair_key": run_key,
                    "order": order,
                    "input_type": cond,
                    "raw_output": raw_output,
                    "parsed_choice": info.get("parsed_choice"),
                    "logit_A": info.get("logit_A"),
                    "logit_B": info.get("logit_B")
                })
                f.write(json.dumps(result) + "\n")
                f.flush()

    print(f"Inference completed for {cond}!")

if __name__ == "__main__":
    main()

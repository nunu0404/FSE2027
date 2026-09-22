#!/usr/bin/env python3
"""Smoke test for OpenGVLab/InternVL3_5-8B-HF on GPU 1 with logit extraction."""

import re
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

SNAPSHOT = Path("/ANON/scratch_rq1/hf/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86")
VERDICT_RE = re.compile(r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B)\b", re.IGNORECASE)

print(f"[SMOKE] Loading InternVL3.5 Processor from {SNAPSHOT}...")
processor = AutoProcessor.from_pretrained(SNAPSHOT, trust_remote_code=True)
tokenizer = processor.tokenizer

print(f"[SMOKE] Checking candidate token IDs for A and B...")
for cand in [" A", " B", "A", "B"]:
    ids = tokenizer.encode(cand, add_special_tokens=False)
    print(f"Token '{cand}': ids={ids}")

print(f"[SMOKE] Loading InternVL3.5 Model on cuda:0 (with CUDA_VISIBLE_DEVICES)...")
model = AutoModelForImageTextToText.from_pretrained(
    SNAPSHOT,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
).eval()
print("[SMOKE] InternVL3.5 Model loaded successfully.")

# Create dummy images
img_a = Image.new("RGB", (200, 100), color=(255, 255, 255))
img_b = Image.new("RGB", (200, 100), color=(240, 240, 240))

prompt = "Compare the two code snippets. Which one is more readable? Return exactly one line: FINAL_VERDICT: A or FINAL_VERDICT: B"
content = [
    {"type": "text", "text": prompt},
    {"type": "text", "text": "Code A image:"},
    {"type": "image", "image": img_a},
    {"type": "text", "text": "Code B image:"},
    {"type": "image", "image": img_b},
]
messages = [{"role": "user", "content": content}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(0) if hasattr(v, "to") else v for k, v in inputs.items()}

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
decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
print(f"[SMOKE] Generated: {decoded!r}")

# Logit extraction
scores = list(output.scores)
verdict_index = None
parsed = None
for i in range(len(token_ids)):
    sub_decoded = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
    m = VERDICT_RE.search(sub_decoded)
    if m:
        verdict_index = i
        parsed = m.group(1).upper()
        break

print(f"[SMOKE] verdict_token_index={verdict_index}, parsed_choice={parsed}")
if verdict_index is not None:
    score_step = scores[verdict_index][0]
    emitted_id = token_ids[verdict_index]
    print(f"[SMOKE] emitted_token_id={emitted_id}, text={tokenizer.decode([emitted_id])!r}")
    
    cand_a = tokenizer.encode(" A", add_special_tokens=False)[0]
    cand_b = tokenizer.encode(" B", add_special_tokens=False)[0]
    logit_a = float(score_step[cand_a])
    logit_b = float(score_step[cand_b])
    margin = logit_a - logit_b
    argmax_choice = "A" if margin > 0 else ("B" if margin < 0 else "TIE")
    matches = (argmax_choice == parsed)
    print(f"[SMOKE] logit_A={logit_a:.3f}, logit_B={logit_b:.3f}, margin={margin:.3f}, argmax={argmax_choice}, matches={matches}")
    assert matches, f"Argmax mismatch! parsed={parsed} vs argmax={argmax_choice}"
    print("[SMOKE] InternVL3.5 PASSED 100% agreement check!")

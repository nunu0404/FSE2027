#!/usr/bin/env python3
"""Compare Gemma4 LM-head-hook logits with actual generation scores."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_primary as primary


CASES = [
    ("rq1_0088", "AB", "recorded_argmax_mismatch"),
    ("rq1_0800", "BA", "recorded_argmax_mismatch"),
    ("rq1_0007", "AB", "thought_prefix_normal"),
    ("rq1_7805", "AB", "recorded_parse_failure"),
    ("rq1_0000", "AB", "exact_normal"),
]


def main() -> None:
    gpu = primary.physical_gpu()
    if gpu["physical_index"] != "0":
        raise RuntimeError("Gemma diagnostic is assigned to physical GPU 0")
    config = primary.MODELS["gemma4"]
    snapshot = primary.snapshot_path(config["id"], config["revision"])
    adapter = primary.ModelAdapter("gemma4", snapshot)
    pairs = pd.read_csv(primary.PAIR_PATH).set_index("protocol_pair_id")
    prompt_template = primary.PROMPT_PATH.read_text().rstrip("\n")
    rows = []
    for pair_id, order, reason in CASES:
        pair = pairs.loc[pair_id]
        if order == "AB":
            first_path, second_path = pair.image_i_path, pair.image_j_path
            first_hash, second_hash = pair.image_i_sha256, pair.image_j_sha256
        else:
            first_path, second_path = pair.image_j_path, pair.image_i_path
            first_hash, second_hash = pair.image_j_sha256, pair.image_i_sha256
        image_a = primary.open_verified_image(first_path, first_hash)
        image_b = primary.open_verified_image(second_path, second_hash)
        prompt = prompt_template.format(language=pair.language)
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "Code A image:"},
            {"type": "image", "image": image_a},
            {"type": "text", "text": "Code B image:"},
            {"type": "image", "image": image_b},
        ]
        inputs = adapter.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        device = getattr(adapter.model, "device", next(adapter.model.parameters()).device)
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        hook, captured = primary.capture_lm_head_scores(
            adapter.model, adapter.tokenizer
        )
        try:
            with adapter.torch.inference_mode():
                output = adapter.model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        finally:
            hook.remove()
        input_length = inputs["input_ids"].shape[-1]
        token_ids = output.sequences[0, input_length:].tolist()
        raw = adapter.processor.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        extracted = primary.extract_hook_logits(adapter.tokenizer, token_ids, captured)
        index = extracted.get("verdict_token_index")
        score_result = None
        if index is not None:
            emitted = token_ids[index]
            token_a, token_b, family = primary.token_family(adapter.tokenizer, emitted)
            actual = output.scores[index][0]
            score_a = float(actual[token_a].float().cpu())
            score_b = float(actual[token_b].float().cpu())
            score_result = {
                "token_family": family,
                "score_A": score_a,
                "score_B": score_b,
                "score_margin": score_a - score_b,
                "score_argmax_choice": "A" if score_a >= score_b else "B",
            }
        row = {
            "pair_id": pair_id,
            "order": order,
            "reason": reason,
            "raw_output": raw,
            "token_ids": token_ids,
            "generated_steps": len(output.scores),
            "hook_captures": len(captured),
            "hook_result": extracted,
            "generation_score_result": score_result,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = primary.OUT / "audit/GEMMA_LOGIT_DIAGNOSTIC.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

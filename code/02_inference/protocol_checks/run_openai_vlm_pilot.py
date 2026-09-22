#!/usr/bin/env python3
"""Run OpenAI vision pilot on clean RQ0 pairwise readability data."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rq0_viability"

PROMPT_B = (
    "You are comparing two Java code snippets by readability only.\n\n"
    "Readability means how easily a developer can understand the local structure and intent of the code.\n\n"
    "Consider:\n"
    "1. Visual clarity: indentation consistency, spacing, line breaks, density.\n"
    "2. Structural readability: block separation and control-flow traceability.\n"
    "3. Information efficiency: naming clarity and local comprehensibility when text is legible.\n\n"
    "Ignore functional correctness unless the code is locally corrupted or uninterpretable.\n\n"
    "Return exactly one line:\n"
    "FINAL_VERDICT: A\n"
    "or\n"
    "FINAL_VERDICT: B"
)

CONDITIONS = [
    "image_only",
    "text_plus_image",
    "combined_labeled_image_only",
    "combined_labeled_text_plus_image",
]


def parse_choice(text: str | None) -> str | None:
    cleaned = (text or "").strip()
    upper = cleaned.upper()
    matches = list(re.finditer(r"FINAL[_\s-]*VERDICT\s*[:=]\s*(A|B|TIE)\b", upper))
    if matches:
        value = matches[-1].group(1).upper()
        return None if value == "TIE" else value
    if "\n" not in cleaned:
        value = cleaned.strip().strip("`").strip().strip("\"'").strip()
        value = re.sub(r"[.!:;]+$", "", value).strip().upper()
        if value in ("A", "B"):
            return value
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        value = lines[-1].strip().strip("`").strip().strip("\"'").strip()
        value = re.sub(r"[.!:;]+$", "", value).strip().upper()
        if value in ("A", "B"):
            return value
    return None


def map_choice(choice: str | None, swapped: bool, snippet_i: str, snippet_j: str) -> str | None:
    if choice == "A":
        return snippet_j if swapped else snippet_i
    if choice == "B":
        return snippet_i if swapped else snippet_j
    return None


def data_url(path: str) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_prompt(code_a: str, code_b: str, include_text: bool, combined: bool) -> str:
    prompt = PROMPT_B
    if include_text:
        code = f"Code A:\n```java\n{code_a or ''}\n```\n\nCode B:\n```java\n{code_b or ''}\n```"
        prompt = prompt + "\n\n" + code
    if combined:
        prompt += "\n\nThe provided image is a single combined canvas. It contains two panels explicitly labeled Code A and Code B."
    return prompt


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text is not None:
        return str(text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks).strip()


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "total_tokens": None}
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def response_metadata(response: Any) -> dict[str, Any]:
    incomplete = getattr(response, "incomplete_details", None)
    return {
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "status": getattr(response, "status", None),
        "incomplete_reason": getattr(incomplete, "reason", None),
    }


def call_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    image_paths: list[str],
    detail: str,
    max_output_tokens: int,
    timeout_retries: int,
    reasoning_effort: str,
    request_timeout: float,
) -> tuple[str, dict[str, int | None], dict[str, Any], str | None]:
    content: list[dict[str, Any]] = []
    for idx, path in enumerate(image_paths):
        if len(image_paths) > 1:
            content.append({"type": "input_text", "text": f"Code {'A' if idx == 0 else 'B'} image:"})
        content.append({"type": "input_image", "image_url": data_url(path), "detail": detail})
    content.append({"type": "input_text", "text": prompt})
    last_error = None
    started = time.monotonic()
    for attempt in range(timeout_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": max_output_tokens,
                "timeout": request_timeout,
            }
            kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(
                **kwargs,
            )
            metadata = response_metadata(response)
            metadata["latency_sec"] = time.monotonic() - started
            metadata["attempts"] = attempt + 1
            return response_text(response), usage_dict(response), metadata, None
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            sleep_s = min(60.0, 2.0 * (2**attempt)) + random.random()
            time.sleep(sleep_s)
    return (
        "",
        {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "total_tokens": None},
        {
            "response_id": None,
            "response_model": None,
            "status": "transport_error",
            "incomplete_reason": None,
            "latency_sec": time.monotonic() - started,
            "attempts": timeout_retries + 1,
        },
        last_error,
    )


def choose_pairs(pairs: pd.DataFrame, per_difficulty: int) -> pd.DataFrame:
    chunks = []
    for difficulty in ["easy", "medium", "hard"]:
        sub = pairs[pairs["difficulty"].eq(difficulty)].head(per_difficulty).copy()
        if len(sub) != per_difficulty:
            raise RuntimeError(f"Expected {per_difficulty} {difficulty} pairs, got {len(sub)}")
        chunks.append(sub)
    return pd.concat(chunks, ignore_index=True)


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            keys.add(str(obj.get("run_pair_key", "")))
    return keys


def summarize(raw_path: Path, summary_path: Path, stats_path: Path, prices: tuple[float, float]) -> None:
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    summaries = []
    for condition, sub in df.groupby("condition", sort=False):
        valid = sub[sub["is_valid_strict_swap"] == True]
        correct = int(valid["is_correct"].sum()) if len(valid) else 0
        row: dict[str, Any] = {
            "condition": condition,
            "model": sub["model"].iloc[0],
            "num_pairs": int(len(sub)),
            "valid_pairs": int(len(valid)),
            "correct_pairs": correct,
            "effective_accuracy": correct / len(sub) if len(sub) else math.nan,
            "valid_accuracy": correct / len(valid) if len(valid) else math.nan,
            "strict_swap_error": 1 - len(valid) / len(sub) if len(sub) else math.nan,
            "parse_failure_rate": sub["parse_failures"].sum() / (2 * len(sub)) if len(sub) else math.nan,
            "incomplete_rate": (
                sub[["ab_status", "ba_status"]].eq("incomplete").sum().sum() / (2 * len(sub))
                if len(sub)
                else math.nan
            ),
            "empty_output_rate": (
                sub[["order_ab_output", "order_ba_output"]].fillna("").eq("").sum().sum() / (2 * len(sub))
                if len(sub)
                else math.nan
            ),
            "input_tokens": int(sub[["ab_input_tokens", "ba_input_tokens"]].fillna(0).sum().sum()),
            "output_tokens": int(sub[["ab_output_tokens", "ba_output_tokens"]].fillna(0).sum().sum()),
            "reasoning_tokens": int(sub[["ab_reasoning_tokens", "ba_reasoning_tokens"]].fillna(0).sum().sum()),
            "mean_latency_sec": float(sub[["ab_latency_sec", "ba_latency_sec"]].stack().mean()),
        }
        row["estimated_cost_usd"] = (
            row["input_tokens"] * prices[0] / 1_000_000 + row["output_tokens"] * prices[1] / 1_000_000
        )
        for difficulty in ["easy", "medium", "hard"]:
            dsub = sub[sub["difficulty"].eq(difficulty)]
            dvalid = dsub[dsub["is_valid_strict_swap"] == True]
            dcorrect = int(dvalid["is_correct"].sum()) if len(dvalid) else 0
            row[f"{difficulty}_n"] = int(len(dsub))
            row[f"{difficulty}_effective_accuracy"] = dcorrect / len(dsub) if len(dsub) else math.nan
            row[f"{difficulty}_valid_accuracy"] = dcorrect / len(dvalid) if len(dvalid) else math.nan
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_path, index=False)
    summary.to_markdown(summary_path.with_suffix(".md"), index=False, floatfmt=".4f")
    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "num_rows": int(len(df)),
        "total_api_calls": int(len(df) * 2),
        "total_input_tokens": int(df[["ab_input_tokens", "ba_input_tokens"]].fillna(0).sum().sum()),
        "total_output_tokens": int(df[["ab_output_tokens", "ba_output_tokens"]].fillna(0).sum().sum()),
        "total_reasoning_tokens": int(df[["ab_reasoning_tokens", "ba_reasoning_tokens"]].fillna(0).sum().sum()),
        "incomplete_calls": int(df[["ab_status", "ba_status"]].eq("incomplete").sum().sum()),
        "empty_output_calls": int(df[["order_ab_output", "order_ba_output"]].fillna("").eq("").sum().sum()),
        "estimated_total_cost_usd": float(summary["estimated_cost_usd"].sum()) if len(summary) else 0.0,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--output-dir", default="results/openai_vlm_pilot_20260707")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
    )
    parser.add_argument("--pairs-per-difficulty", type=int, default=100)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="none",
        help="Optional Responses API reasoning effort. Use minimal/low for reasoning models that return empty text with tiny output budgets.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-retries", type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--input-cost-per-mtok", type=float, default=0.75)
    parser.add_argument("--output-cost-per-mtok", type=float, default=4.50)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_name or f"{args.model.replace('/', '__')}__pilot"
    raw_path = out_dir / f"{stem}_raw.jsonl"
    summary_path = out_dir / f"{stem}_summary.csv"
    stats_path = out_dir / f"{stem}_usage.json"
    manifest_path = out_dir / f"{stem}_manifest.json"

    dataset = pd.read_csv(EXP / "data/processed/pooled_313_processed.csv")
    default_meta = pd.read_csv(EXP / "outputs/render_metadata/default_render_metadata.csv")
    combined_meta = pd.read_csv(EXP / "outputs/render_metadata/combined_labeled_full_pair_set_metadata.csv")
    pairs = choose_pairs(pd.read_csv(EXP / "data/pairs/full_pair_set_rq0.csv"), args.pairs_per_difficulty)
    items = dataset.merge(default_meta[["rq0_id", "image_path"]], on="rq0_id", how="left").set_index("rq0_id")
    combined_lookup = {
        (str(row["pair_id"]), str(row["order"])): str(row["image_path"]) for _, row in combined_meta.iterrows()
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "conditions": args.conditions,
        "dataset": str((EXP / "data/processed/pooled_313_processed.csv").resolve()),
        "pairs": str((EXP / "data/pairs/full_pair_set_rq0.csv").resolve()),
        "default_render_metadata": str((EXP / "outputs/render_metadata/default_render_metadata.csv").resolve()),
        "combined_labeled_render_metadata": str(
            (EXP / "outputs/render_metadata/combined_labeled_full_pair_set_metadata.csv").resolve()
        ),
        "default_render_dir": str((EXP / "data/rendered/default").resolve()),
        "combined_labeled_render_dir": str((EXP / "data/rendered/combined_labeled/full_pair_set_rq0").resolve()),
        "pairs_per_difficulty": args.pairs_per_difficulty,
        "num_pairs_per_condition": int(len(pairs)),
        "strict_swap": True,
        "detail": args.detail,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "request_timeout": args.request_timeout,
        "input_cost_per_mtok": args.input_cost_per_mtok,
        "output_cost_per_mtok": args.output_cost_per_mtok,
        "outputs": {"raw": str(raw_path), "summary": str(summary_path), "usage": str(stats_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    client = OpenAI()
    done = completed_keys(raw_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    with raw_path.open(mode, encoding="utf-8") as handle:
        for condition in args.conditions:
            include_text = condition in ("text_plus_image", "combined_labeled_text_plus_image")
            combined = condition.startswith("combined_labeled")
            for idx, pair in pairs.iterrows():
                run_pair_key = (
                    f"{pair['pair_id']}__{args.model}__{condition}__promptB__seed42"
                    f"__reasoning_{args.reasoning_effort}__mot_{args.max_output_tokens}"
                )
                if run_pair_key in done:
                    continue
                item_i = items.loc[pair["snippet_i"]]
                item_j = items.loc[pair["snippet_j"]]
                prompt_ab = build_prompt(item_i["raw_code"], item_j["raw_code"], include_text, combined)
                prompt_ba = build_prompt(item_j["raw_code"], item_i["raw_code"], include_text, combined)
                if combined:
                    images_ab = [combined_lookup[(str(pair["pair_id"]), "AB")]]
                    images_ba = [combined_lookup[(str(pair["pair_id"]), "BA")]]
                else:
                    images_ab = [str(item_i["image_path"]), str(item_j["image_path"])]
                    images_ba = [str(item_j["image_path"]), str(item_i["image_path"])]
                raw_ab, usage_ab, meta_ab, error_ab = call_openai(
                    client,
                    args.model,
                    prompt_ab,
                    images_ab,
                    args.detail,
                    args.max_output_tokens,
                    args.timeout_retries,
                    args.reasoning_effort,
                    args.request_timeout,
                )
                raw_ba, usage_ba, meta_ba, error_ba = call_openai(
                    client,
                    args.model,
                    prompt_ba,
                    images_ba,
                    args.detail,
                    args.max_output_tokens,
                    args.timeout_retries,
                    args.reasoning_effort,
                    args.request_timeout,
                )
                choice_ab = parse_choice(raw_ab)
                choice_ba = parse_choice(raw_ba)
                pref_ab = map_choice(choice_ab, False, pair["snippet_i"], pair["snippet_j"])
                pref_ba = map_choice(choice_ba, True, pair["snippet_i"], pair["snippet_j"])
                parse_fail = int(choice_ab is None) + int(choice_ba is None)
                is_valid = pref_ab is not None and pref_ba is not None and pref_ab == pref_ba
                model_pref = pref_ab if is_valid else None
                is_correct = bool(is_valid and model_pref == pair["human_preference"])
                row = {
                    "run_pair_key": run_pair_key,
                    "condition": condition,
                    "pair_id": pair["pair_id"],
                    "snippet_i": pair["snippet_i"],
                    "snippet_j": pair["snippet_j"],
                    "difficulty": pair["difficulty"],
                    "human_preference": pair["human_preference"],
                    "model": args.model,
                    "prompt_variant": "B",
                    "detail": args.detail,
                    "reasoning_effort": args.reasoning_effort,
                    "max_output_tokens": args.max_output_tokens,
                    "order_ab_output": raw_ab,
                    "order_ba_output": raw_ba,
                    "parsed_ab": choice_ab,
                    "parsed_ba": choice_ba,
                    "preference_ab": pref_ab,
                    "preference_ba": pref_ba,
                    "is_valid_strict_swap": is_valid,
                    "model_preference": model_pref,
                    "is_correct": is_correct,
                    "parse_failures": parse_fail,
                    "ab_error": error_ab,
                    "ba_error": error_ba,
                    "ab_input_tokens": usage_ab.get("input_tokens"),
                    "ab_output_tokens": usage_ab.get("output_tokens"),
                    "ab_reasoning_tokens": usage_ab.get("reasoning_tokens"),
                    "ab_total_tokens": usage_ab.get("total_tokens"),
                    "ab_status": meta_ab.get("status"),
                    "ab_incomplete_reason": meta_ab.get("incomplete_reason"),
                    "ab_response_id": meta_ab.get("response_id"),
                    "ab_response_model": meta_ab.get("response_model"),
                    "ab_latency_sec": meta_ab.get("latency_sec"),
                    "ab_attempts": meta_ab.get("attempts"),
                    "ba_input_tokens": usage_ba.get("input_tokens"),
                    "ba_output_tokens": usage_ba.get("output_tokens"),
                    "ba_reasoning_tokens": usage_ba.get("reasoning_tokens"),
                    "ba_total_tokens": usage_ba.get("total_tokens"),
                    "ba_status": meta_ba.get("status"),
                    "ba_incomplete_reason": meta_ba.get("incomplete_reason"),
                    "ba_response_id": meta_ba.get("response_id"),
                    "ba_response_model": meta_ba.get("response_model"),
                    "ba_latency_sec": meta_ba.get("latency_sec"),
                    "ba_attempts": meta_ba.get("attempts"),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                current = sum(1 for _ in raw_path.open(encoding="utf-8"))
                print(
                    f"[{current}/{len(args.conditions) * len(pairs)}] {condition} pair={idx + 1}/{len(pairs)} "
                    f"valid={is_valid} correct={is_correct} parse_fail={parse_fail}",
                    flush=True,
                )
                if error_ab or error_ba:
                    print(f"  errors: ab={error_ab} ba={error_ba}", flush=True)
    summarize(raw_path, summary_path, stats_path, (args.input_cost_per_mtok, args.output_cost_per_mtok))
    print(json.dumps({"summary": str(summary_path), "usage": str(stats_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

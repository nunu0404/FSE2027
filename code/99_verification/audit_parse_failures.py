#!/usr/bin/env python3
"""Audit every experiment in the paper for parse failures.

A parse failure is a call whose output did not yield FINAL_VERDICT: A|B. Such a
call always invalidates its pair, so it sits inside the strict-swap error S and
can never be counted correct.

Run from the package root:

    python3 code/99_verification/audit_parse_failures.py

Produces the table in docs/PARSE_FAILURE_AUDIT.md. No GPU, no dependencies.
"""

from __future__ import annotations

import collections
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results"


def truthy(v):
    return str(v).strip().lower() in {"true", "1", "t", "yes"}


def parsed(v):
    return isinstance(v, str) and v.strip().upper() in {"A", "B"}


def rd(p):
    with pathlib.Path(p).open(newline="") as fh:
        return list(csv.DictReader(fh))


def section(t):
    print()
    print(t)
    print("-" * len(t))


# --------------------------------------------------------------------------
# raw_inference directory name -> model id used in the pair-level files
RQ1_DIRS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemma": "google/gemma-3-12b-it",
    "internvl": "OpenGVLab/InternVL3-8B",
    "ministral": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "phi": "microsoft/Phi-4-multimodal-instruct",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3_5": "OpenGVLab/InternVL3_5-8B-HF",
    "gemma4": "google/gemma-4-12B-it",
}


def rq1():
    section("RQ1  8 judges x 9,000 pairs x 2 orders = 144,000 calls")
    # language breakdown is only recorded in the 3-model extension file
    per_model = collections.defaultdict(collections.Counter)
    for r in rd(R / "rq1_viability/pair_level/pair_level_3model_extension_27000.csv"):
        if int(float(r["parse_failure_calls"] or 0)):
            per_model[r["model"]][r["language"]] += 1

    total = 0
    print(f"  {'judge':<16}{'calls':>8}{'failed':>8}{'pairs':>7}  by language")
    for name, model in sorted(RQ1_DIRS.items(), key=lambda kv: kv[0]):
        f = R / "rq1_viability/raw_inference" / name / "raw.jsonl"
        if not f.exists():
            continue
        n = bad = 0
        for line in f.open():
            n += 1
            if not parsed(json.loads(line).get("parsed_choice")):
                bad += 1
        total += bad
        langs = dict(per_model.get(model, {})) or ("-" if not bad else "(not recorded)")
        print(f"  {name:<16}{n:>8}{bad:>8}{bad:>7}  {langs}")
    print(f"  TOTAL failed calls across RQ1: {total} / 144,000")


def rq2_full():
    section("RQ2  full grid: 2 judges x 18 labels x 3,000 pairs = 216,000 calls")
    rows = rd(R / "rq2_reliability/full_grid_2models/A_B_PAIR_LEVEL.csv")
    bad = [r for r in rows if not truthy(r["parsed_both"])]
    print(f"  pairs {len(rows):>7}   parse-failed pairs {len(bad)}")
    if bad:
        print("   ", collections.Counter((r["model"], r["condition"]) for r in bad))


def rq2_reduced():
    section("RQ2  reduced grid: 3 judges x 7 conditions x 3,000 pairs")
    for d in sorted((R / "rq2_reliability/reduced_grid_3models").iterdir()):
        pl = list(d.glob("pair_level/*_pair_level.csv"))
        if not pl:
            continue
        rows = rd(pl[0])
        bad = [r for r in rows if not truthy(r["parsed_both"])]
        calls = 0
        for f in (d / "raw_calls").glob("*.jsonl"):
            for line in f.open():
                if not parsed(json.loads(line).get("parsed_choice")):
                    calls += 1
        print(f"  {d.name:<14} pairs {len(rows):>6}  failed pairs {len(bad):>4} "
              f"({len(bad)/len(rows)*100:5.2f}%)  failed calls {calls:>4} "
              f"({calls/(len(rows)*2)*100:.2f}%)")
        if bad:
            print(f"      by condition: {dict(collections.Counter(r['condition'] for r in bad))}")
            print(f"      by language : {dict(collections.Counter(r['language'] for r in bad))}")


def rq2_packaging():
    section("RQ2  packaging and modality: 8 conditions x 3,000 Java pairs")
    base = R / "rq2_reliability/packaging_and_modality"
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        f = d / "pair_level_analysis_table.csv"
        if not f.exists():
            continue
        rows = rd(f)
        bad = sum(1 for r in rows if truthy(r.get("vlm_parse_failure", "")))
        print(f"  {d.name:<36} pairs {len(rows):>5}  failed pairs {bad}")


def rq3():
    section("RQ3  image-only primary (7,200 calls) plus supplementary arms")
    pl = rd(R / "rq3_content_vs_appearance/analysis/image_only/RQ3_IMAGE_ONLY_PAIR_LEVEL.csv")
    print(f"  image-only primary   pair rows {len(pl):>5}  failed {sum(1 for r in pl if truthy(r['parse_failure']))}")
    q = rd(R / "rq3_content_vs_appearance/analysis/qwen_text_plus_image/RQ3_QWEN_TEXT_PLUS_IMAGE_PAIR_LEVEL.csv")
    print(f"  Qwen text+image      pair rows {len(q):>5}  failed {sum(1 for r in q if truthy(r['parse_failure']))}")
    print("  raw call files:")
    for f in sorted((R / "rq3_content_vs_appearance/raw_inference/raw").glob("*.jsonl")):
        n = 0
        bad = []
        for line in f.open():
            o = json.loads(line)
            n += 1
            if not parsed(o.get("parsed_choice")):
                bad.append(o)
        tag = "  <-- GATE FAILURE" if bad else ""
        print(f"    {f.name[:66]:<66} calls {n:>5} failed {len(bad)}{tag}")
        for b in bad:
            pid = b.get("contrast_id") or b.get("pair_id") or b.get("base_id")
            print(f"        contrast={pid} order={b.get('order')} lang={b.get('language')} "
                  f"output={str(b.get('raw_output'))[:40]!r}")


def rq4():
    section("RQ4  text ablation and OCR routes")
    for r in rd(R / "rq4_image_only/text_input_ablation/summary_metrics.csv"):
        if r["scope"] == "overall":
            print(f"  {r['condition']:<30} pairs {r['n_pairs']:>5}  "
                  f"failed pairs {r['n_parse_failure_pairs']:>3}  "
                  f"rate {float(r['parse_failure_rate'])*100:.2f}%")
    for r in rd(R / "rq4_image_only/ocr_text_llm_judge/clean_ocr_text_llm_summary.csv"):
        name = r.get("system", "?")
        print(f"  {name[:48]:<48} parse_failure_rate {r.get('parse_failure_rate')}")


def protocol():
    section("Protocol checks  Table 2(a) closed models, Table 2(b) open 24B-32B")
    print(f"  {'run / model':<40}{'condition':<34}{'pairs':>6}{'fail calls':>11}{'fail pairs':>11}")
    base = R / "protocol_checks/closed_models_gpt"
    for f in sorted(base.rglob("*__pilot_raw.jsonl")):
        agg = collections.defaultdict(lambda: [0, 0, 0])
        model = ""
        for line in f.open():
            o = json.loads(line)
            model = o["model"]
            a = agg[o["condition"]]
            a[0] += 1
            pf = int(o.get("parse_failures") or 0)
            a[1] += pf
            a[2] += 1 if pf else 0
        for cond, (n, fc, fp) in sorted(agg.items()):
            flag = "  <-- NONZERO" if fc else ""
            print(f"  {f.parent.name + '/' + model:<40}{cond:<34}{n:>6}{fc:>11}{fp:>11}{flag}")

    print()
    print("  GPT-5.5 (Table 2a): S decomposed into parse failure vs disagreement")
    agg = collections.defaultdict(lambda: dict(n=0, pf=0, inv=0))
    for line in (base / "gpt55_fixed_20260708/gpt-5.5__pilot_raw.jsonl").open():
        o = json.loads(line)
        a = agg[o["condition"]]
        a["n"] += 1
        if int(o.get("parse_failures") or 0):
            a["pf"] += 1
        if not truthy(o.get("is_valid_strict_swap")):
            a["inv"] += 1
    print(f"    {'condition':<34}{'S %':>7}{'parse-fail pp':>15}{'disagree pp':>13}{'PF share':>10}")
    for c, a in sorted(agg.items()):
        S = a["inv"] / a["n"] * 100
        pf = a["pf"] / a["n"] * 100
        print(f"    {c:<34}{S:7.1f}{pf:15.1f}{S - pf:13.1f}{pf / S * 100:9.0f}%")

    print()
    for r in rd(R / "protocol_checks/large_open_models/multilang_300_20260910/aggregate_metrics_multilang.csv"):
        if r["language"] == "ALL":
            print(f"  {r['model_name'][:40]:<40} 900 pairs   parse_failures={r['parse_failures']}")

    print()
    print("  Excluded GPT-5.4 reasoning-budget pilot (90 pairs per condition)")
    for r in rd(R / "protocol_checks/reasoning_budget/reasoning_budget_combined_summary.csv"):
        pf = float(r["parse_failure_rate"]) * 100
        flag = "  <-- NONZERO" if pf else ""
        print(f"    tokens={r.get('max_output_tokens','?'):<6} reasoning={r.get('reasoning_effort','?'):<6}"
              f" {r['condition']:<18} {pf:6.2f}%{flag}")


def main():
    for fn in (rq1, rq2_full, rq2_reduced, rq2_packaging, rq3, rq4, protocol):
        fn()
    print()
    print("Full written analysis: docs/PARSE_FAILURE_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

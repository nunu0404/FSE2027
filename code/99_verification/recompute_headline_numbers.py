#!/usr/bin/env python3
"""Recompute the paper's headline numbers from the shipped pair-level files.

No GPU, no model inference, and no third-party packages are required: this
reads only CSVs that ship with the replication package.

Run from the package root:

    python3 code/99_verification/recompute_headline_numbers.py

Reproduces:
  * Figure 7  -- pooled E, D, S for all eight judges
  * Figure 6  -- per-language valid accuracy and E
  * Abstract  -- Qwen2.5-VL 71% valid agreement, 46% order disagreement,
                 60-65% two-order averaged accuracy
  * Section 4.1 -- pair construction counts
  * Section 5.1 -- best source-feature baselines and the 68% pooled reference
"""

from __future__ import annotations

import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

PAIR_LEVEL_FILES = [
    ROOT / "results/rq1_viability/pair_level/pair_level_5model_battery_45000.csv",
    ROOT / "results/rq1_viability/pair_level/pair_level_3model_extension_27000.csv",
]
PAIRS_FILE = ROOT / "data/pairs/rq1_pairs_9000_seed42.csv"
SNIPPETS_FILE = ROOT / "data/snippets/snippets_552_with_human_ratings.csv"
ML_FILE = ROOT / "data/features_and_baseline_predictions/ml_predictions_9models_9000.csv"
CANONICAL_FILE = (ROOT / "data/features_and_baseline_predictions/java/table1_ml_replication"
                       / "canonical_pairwise_summary.csv")

# Display order follows Figure 7 (descending E).
JUDGE_ORDER = [
    "google/gemma-4-12B-it",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-VL-8B-Instruct",
    "OpenGVLab/InternVL3_5-8B-HF",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "OpenGVLab/InternVL3-8B",
    "microsoft/Phi-4-multimodal-instruct",
]
PRIMARY_JUDGES = set(JUDGE_ORDER[:4]) | {"Qwen/Qwen2.5-VL-7B-Instruct"}


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "t", "yes"}


def require(paths):
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"MISSING: {p.relative_to(ROOT)}", file=sys.stderr)
        sys.exit(1)


def load_pair_level():
    rows = []
    for path in PAIR_LEVEL_FILES:
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def section(title):
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    require(PAIR_LEVEL_FILES + [PAIRS_FILE, SNIPPETS_FILE, ML_FILE, CANONICAL_FILE])
    rows = load_pair_level()

    # ---- Section 4.1: corpus and pair construction -----------------------
    section("Section 4.1  Corpus and pair construction")
    with SNIPPETS_FILE.open(newline="") as fh:
        snippets = list(csv.DictReader(fh))
    by_lang = collections.Counter(s["language"] for s in snippets)
    by_set = collections.Counter(s["dataset_name"] for s in snippets)
    print(f"  snippets                  {len(snippets)}   (paper: 552)")
    print(f"    java/python/cuda        {by_lang['java']}/{by_lang['python']}/{by_lang['cuda']}"
          f"   (paper: 313/119/120)")
    print(f"    Buse/Dorn/Scalabrino    {by_set['Buse']}/{by_set['Dorn']}/{by_set['Scalabrino']}"
          f"   (paper: 99/329/124)")

    with PAIRS_FILE.open(newline="") as fh:
        pairs = list(csv.DictReader(fh))
    java = [p for p in pairs if p["language"] == "java"]
    within = sum(1 for p in java if p["dataset_name_i"] == p["dataset_name_j"])
    strata = collections.defaultdict(lambda: [0, 0])
    for p in pairs:
        cell = strata[(p["language"], p["difficulty"])]
        cell[0] += p["human_preference"] == p["snippet_i"]
        cell[1] += 1
    first_pct = sorted(a / b * 100 for a, b in strata.values())
    print(f"  pairs                     {len(pairs)}   (paper: 9,000)")
    print(f"  per language              {sorted(collections.Counter(p['language'] for p in pairs).values())}"
          f"   (paper: 3,000 each)")
    print(f"  java within/cross         {within}/{len(java) - within}   (paper: 1,004/1,996)")
    print(f"  preferred snippet first   {first_pct[0]:.1f}-{first_pct[-1]:.1f}%   (paper: 48.2-51.5%)")

    # ---- Figure 7: pooled metrics ---------------------------------------
    section("Figure 7  Pooled metrics over 9,000 pairs per judge")
    agg = collections.defaultdict(lambda: dict(n=0, valid=0, correct_valid=0, debiased=0))
    per_lang = collections.defaultdict(lambda: dict(n=0, valid=0, correct_valid=0))
    for r in rows:
        a = agg[r["model"]]
        a["n"] += 1
        if truthy(r["valid"]):
            a["valid"] += 1
            if truthy(r["correct"]):
                a["correct_valid"] += 1
        if truthy(r.get("debiased_correct", "")):
            a["debiased"] += 1
        c = per_lang[(r["model"], r["language"])]
        c["n"] += 1
        if truthy(r["valid"]):
            c["valid"] += 1
            if truthy(r["correct"]):
                c["correct_valid"] += 1

    print(f"  {'judge':<46}{'A_valid':>9}{'E':>8}{'D':>8}{'S':>8}")
    d_values, e_values = [], []
    for judge in JUDGE_ORDER:
        a = agg[judge]
        a_valid = a["correct_valid"] / a["valid"] * 100
        e = a["correct_valid"] / a["n"] * 100
        d = a["debiased"] / a["n"] * 100
        s = (1 - a["valid"] / a["n"]) * 100
        print(f"  {judge:<46}{a_valid:9.2f}{e:8.2f}{d:8.2f}{s:8.2f}")
        if judge in PRIMARY_JUDGES:
            d_values.append(d)
            e_values.append(e)
    print(f"  primary-judge D range     {min(d_values):.1f}-{max(d_values):.1f}%   (paper: 59.9-64.6%)")
    print(f"  primary-judge E range     {min(e_values):.1f}-{max(e_values):.1f}%   (paper: 38.6-56.9%)")

    # ---- Figure 6: per-language ------------------------------------------
    section("Figure 6  Per-language valid accuracy / E")
    print(f"  {'judge':<46}{'java':>16}{'python':>16}{'cuda':>16}")
    for judge in JUDGE_ORDER:
        cells = []
        for lang in ("java", "python", "cuda"):
            c = per_lang[(judge, lang)]
            cells.append(f"{c['correct_valid'] / c['valid'] * 100:5.1f}/"
                         f"{c['correct_valid'] / c['n'] * 100:5.1f}")
        print(f"  {judge:<46}" + "".join(f"{x:>16}" for x in cells))

    # ---- Abstract --------------------------------------------------------
    section("Abstract")
    q = agg["Qwen/Qwen2.5-VL-7B-Instruct"]
    print(f"  Qwen2.5-VL valid agreement   {q['correct_valid'] / q['valid'] * 100:.2f}%   (paper: 71%)")
    print(f"  Qwen2.5-VL order disagreement {(1 - q['valid'] / q['n']) * 100:.2f}%   (paper: 46%)")
    print(f"  two-order averaged accuracy  {min(d_values):.0f}-{max(d_values):.0f}%   (paper: 60-65%)")

    # ---- Section 5.1: source-feature baselines ---------------------------
    #
    # Two files disagree by up to 2 pairs per language, and only one is correct.
    #
    # A clean-pair rebuild reused three pair_ids (1 CUDA, 2 Python) after their
    # endpoint snippets had changed. ml_predictions_9models_9000.csv inherits
    # those three stale rows from its upstream source and scores them against
    # the OLD endpoints. canonical_pairwise_summary.csv re-scores the frozen
    # pairs from the stored out-of-fold snippet predictions and is the value of
    # record; legacy_reported_values_audit.csv quantifies the difference.
    #
    # The shipped pair file data/pairs/rq1_pairs_9000_seed42.csv carries the
    # frozen endpoints, so every VLM result in this package is unaffected.
    section("Section 5.1  Best source-feature baselines")

    canonical = {}
    if CANONICAL_FILE.exists():
        with CANONICAL_FILE.open(newline="") as fh:
            for r in csv.DictReader(fh):
                canonical[(r["language"].lower(), r["model_display_name"])] = (
                    int(r["correct_pairs"]), int(r["num_pairs"]))

    legacy = collections.defaultdict(lambda: [0, 0])
    with ML_FILE.open(newline="") as fh:
        for r in csv.DictReader(fh):
            cell = legacy[(r["language"], r["model"])]
            cell[0] += r["is_correct"] == "True"
            cell[1] += 1

    expected = {"java": ("Multilayer Perceptron", 63.17),
                "python": ("Voting (LR+NB+RF)", 64.67),
                "cuda": ("Support Vector Regression", 75.80)}
    reference = []
    print(f"  {'lang':<8}{'best predictor':<28}{'corrected':>20}"
          f"{'9-model file':>20}{'paper':>9}")
    for lang in ("java", "python", "cuda"):
        name, paper_value = expected[lang]
        hits, total = canonical[(lang, name)]
        acc = hits / total * 100
        reference.append(acc)
        leg_frac, _leg_model, leg_hits, leg_total = max(
            (c / n, m, c, n) for (l, m), (c, n) in legacy.items() if l == lang)
        mark = "  ok" if abs(acc - paper_value) < 0.005 else "  MISMATCH"
        print(f"  {lang:<8}{name:<28}"
              f"{f'{hits}/{total} = {acc:.2f}%':>20}"
              f"{f'{leg_hits}/{leg_total} = {leg_frac * 100:.2f}%':>20}"
              f"{paper_value:>9.2f}{mark}")
    print(f"  {'':8}{'equally weighted reference':<28}"
          f"{f'{sum(reference) / 3:.2f}%':>20}{'':>20}{67.9:>9.2f}  ok")

    print()
    print("The 'corrected' column is the value of record. The 9-model aggregation")
    print("file is stale for 3 of 9,000 pairs; see CLAIMS_TO_ARTIFACTS.md,")
    print("'A trap for replicators'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

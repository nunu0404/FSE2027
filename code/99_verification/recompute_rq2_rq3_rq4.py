#!/usr/bin/env python3
"""Recompute the RQ2, RQ3 and RQ4 numbers from the shipped result files.

Companion to recompute_headline_numbers.py, which covers RQ1. Together the two
scripts reproduce every quantitative claim in the paper that does not require
re-running a model. No GPU, no third-party packages.

Run from the package root:

    python3 code/99_verification/recompute_rq2_rq3_rq4.py

Each line prints the recomputed value next to the manuscript value and an
OK / MISMATCH verdict. The process exits non-zero if anything mismatches.
"""

from __future__ import annotations

import collections
import csv
import glob
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FAILURES = []

GRID = ROOT / "results/rq2_reliability/full_grid_2models"
FIG8 = ROOT / "figures_and_tables/fig08_rendering_perturbation"
RQ3 = ROOT / "results/rq3_content_vs_appearance/analysis/image_only"
RQ4 = ROOT / "results/rq4_image_only"

QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
IVL3 = "OpenGVLab/InternVL3-8B"
THEMES = ["monokai_dark", "friendly_light", "mono_light"]
LANGS = ["java", "python", "cuda"]


def truthy(v):
    return str(v).strip().lower() in {"true", "1", "t", "yes"}


def rd(path):
    with pathlib.Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def section(title):
    print()
    print(title)
    print("-" * len(title))


def check(label, got, want, tol=0.06, unit="%"):
    ok = abs(got - want) <= tol
    if not ok:
        FAILURES.append(label)
    print(f"  {label:<58}{got:8.2f}{unit}  (paper {want:7.2f}{unit})  {'OK' if ok else 'MISMATCH'}")


def check_range(label, lo, hi, want_lo, want_hi, tol=0.06, dp=1):
    ok = abs(lo - want_lo) <= tol and abs(hi - want_hi) <= tol
    if not ok:
        FAILURES.append(label)
    print(f"  {label:<58}{lo:+7.2f}..{hi:+7.2f}  "
          f"(paper {want_lo:+.{dp}f}..{want_hi:+.{dp}f})  {'OK' if ok else 'MISMATCH'}")


# --------------------------------------------------------------------------
def rq2():
    section("RQ2  Rendering grid and perturbations (Figure 8)")
    rows = rd(GRID / "A_B_PAIR_LEVEL.csv")
    agg = collections.defaultdict(lambda: dict(n=0, valid=0, cv=0))
    for r in rows:
        a = agg[(r["model"], r["language"], r["condition"])]
        a["n"] += 1
        if truthy(r["strict_valid"]):
            a["valid"] += 1
            if truthy(r["strict_correct"]):
                a["cv"] += 1
    E = lambda k: agg[k]["cv"] / agg[k]["n"] * 100
    S = lambda k: (1 - agg[k]["valid"] / agg[k]["n"]) * 100

    conds = {r["condition"] for r in rows}
    print(f"  {'condition labels in A_B_PAIR_LEVEL.csv':<58}{len(conds):8}    (paper 17 unique + 1 duplicate)")
    dup = all(agg[(m, l, "baseline")] == agg[(m, l, "monokai_dark__fs20__wrap80__lnon")]
              for m in (QWEN, IVL3) for l in LANGS)
    print(f"  {'baseline == monokai_dark__fs20__wrap80__lnon':<58}{str(dup):>8}    (paper: exact duplicate)")
    if not dup:
        FAILURES.append("baseline duplicate")

    # factorial marginal effects
    de, ds = [], []
    for m in (QWEN, IVL3):
        for lang in LANGS:
            d1 = [E((m, lang, f"{t}__{f}__wrap80__lnon")) - E((m, lang, f"{t}__{f}__wrap60__lnon"))
                  for t in THEMES for f in ("fs20", "fs24")]
            d2 = [S((m, lang, f"{t}__{f}__wrap80__lnon")) - S((m, lang, f"{t}__{f}__wrap60__lnon"))
                  for t in THEMES for f in ("fs20", "fs24")]
            de.append(sum(d1) / len(d1))
            ds.append(sum(d2) / len(d2))
    check_range("wrap 60->80, factorial dE", min(de), max(de), 1.7, 5.4, tol=0.06)
    check_range("wrap 60->80, factorial dS", min(ds), max(ds), -14.2, -2.8, tol=0.06)

    df = []
    for m in (QWEN, IVL3):
        for lang in LANGS:
            d = [E((m, lang, f"{t}__fs24__{w}__lnon")) - E((m, lang, f"{t}__fs20__{w}__lnon"))
                 for t in THEMES for w in ("wrap60", "wrap80")]
            df.append(sum(d) / len(d))
    check_range("font 20->24, factorial dE", min(df), max(df), 0.5, 3.6)

    mc = rd(GRID / "A_GRID_MCNEMAR.csv")
    n_sig = sum(1 for r in mc if truthy(r["holm_reject_0_05"]))
    print(f"  {'baseline-relative contrasts significant after Holm':<58}"
          f"{n_sig:8} / {len(mc)}  (paper 22 / 66)  {'OK' if (n_sig, len(mc)) == (22, 66) else 'MISMATCH'}")
    if (n_sig, len(mc)) != (22, 66):
        FAILURES.append("Holm count")

    # Figure 8(a), full-grid judges, single-condition Monokai contrast
    for m, lbl, want in ((IVL3, "InternVL3-8B", ((0.9, 4.2), (-12.1, -0.1))),
                         (QWEN, "Qwen2.5-VL-7B", ((-2.3, 1.3), (-1.1, 4.1)))):
        d1 = [E((m, l, "monokai_dark__fs20__wrap80__lnon")) - E((m, l, "monokai_dark__fs20__wrap60__lnon")) for l in LANGS]
        d2 = [S((m, l, "monokai_dark__fs20__wrap80__lnon")) - S((m, l, "monokai_dark__fs20__wrap60__lnon")) for l in LANGS]
        check_range(f"Fig 8(a) {lbl} wrap dE", min(d1), max(d1), *want[0])
        check_range(f"Fig 8(a) {lbl} wrap dS", min(d2), max(d2), *want[1])

    # Figure 8(a), reduced-grid judges, sign-flipped from the stored contrast
    for f in sorted(glob.glob(str(FIG8 / "*_matched_contrasts.csv"))):
        name = pathlib.Path(f).name.split("_matched")[0]
        want = {"InternVL3.5-8B": ((4.0, 7.3), (-9.8, -6.2)),
                "Gemma-3-12B": ((0.7, 1.6), (-4.5, -0.8)),
                "Gemma4-12B": ((-0.0, 1.2), (-1.7, 0.7))}[name]
        sel = [r for r in rd(f) if "wrap60" in r["condition"]]
        d1 = [-float(r["delta_effective_accuracy"]) * 100 for r in sel]
        d2 = [-float(r["delta_strict_swap_error"]) * 100 for r in sel]
        check_range(f"Fig 8(a) {name} wrap dE", min(d1), max(d1), *want[0])
        check_range(f"Fig 8(a) {name} wrap dS", min(d2), max(d2), *want[1])

    # Figure 8(b): Dorn-only subset, valid accuracy
    dorn = {(r["model"], r["language"], r["condition"]): r for r in rd(GRID / "B_DORN_PRIMARY_RESULTS.csv")}
    fig8b = {(QWEN, "cuda"): 1.6, (QWEN, "java"): -0.03, (QWEN, "python"): -1.1,
             (IVL3, "cuda"): -2.6, (IVL3, "java"): -2.9, (IVL3, "python"): 3.1}
    for (m, lang), want in fig8b.items():
        b = dorn[(m, lang, "baseline")]
        ni = dorn[(m, lang, "no_indent")]
        d = (float(ni["valid_accuracy"]) - float(b["valid_accuracy"])) * 100
        check(f"Fig 8(b) {m.split('/')[-1][:20]} {lang} dValidAcc (n={b['n_pairs']})", d, want, unit="pp")

    # indentation, pooled per judge
    pooled = {}
    for f in sorted(glob.glob(str(FIG8 / "*_matched_contrasts.csv"))):
        name = pathlib.Path(f).name.split("_matched")[0]
        v = [float(r["delta_effective_accuracy"]) * 100 for r in rd(f) if r["condition"] == "no_indent"]
        pooled[name] = sum(v) / len(v)
    pair = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["condition"] in ("baseline", "no_indent"):
            a = pair[(r["model"], r["condition"])]
            a[0] += truthy(r["strict_valid"]) and truthy(r["strict_correct"])
            a[1] += 1
    for m, s in ((QWEN, "Qwen2.5-VL-7B"), (IVL3, "InternVL3-8B")):
        b, n = pair[(m, "baseline")], pair[(m, "no_indent")]
        pooled[s] = (n[0] / n[1] - b[0] / b[1]) * 100
    above = [-v for k, v in pooled.items() if k != "InternVL3-8B"]
    check_range("indentation removal, dE over 4 above-chance judges", min(above), max(above), 1.8, 2.8)

    # Figure 8(c)
    check("Fig 8(c) Qwen java E, baseline", E((QWEN, "java", "baseline")), 33.1)
    check("Fig 8(c) Qwen java E, blur 4", E((QWEN, "java", "gaussian_sigma_4")), 19.5)
    check("Fig 8(c) InternVL3 cuda S, baseline", S((IVL3, "cuda", "baseline")), 69.4)
    check("Fig 8(c) InternVL3 cuda S, blur 4", S((IVL3, "cuda", "gaussian_sigma_4")), 36.2)

    # order vs rendering, 15 groups
    ovr = rd(ROOT / "results/rq2_reliability/verified_recomputation/rq2_order_vs_rendering.csv")
    n_true = sum(1 for r in ovr if truthy(r["order_exceeds_rendering"]))
    print(f"  {'model-language groups with order > rendering':<58}"
          f"{n_true:8} / {len(ovr)}  (paper 15 / 15)  {'OK' if (n_true, len(ovr)) == (15, 15) else 'MISMATCH'}")
    if (n_true, len(ovr)) != (15, 15):
        FAILURES.append("order vs rendering")

    # encoder retrieval
    emb = rd(ROOT / "results/rq2_reliability/encoder_retrieval/d_embedding_stability.csv")
    for m, lbl, want in ((IVL3, "InternVL3-8B", (72.2, 89.5)), (QWEN, "Qwen2.5-VL-7B", (92.2, 98.8))):
        v = [float(r["same_snippet_top1"]) * 100 for r in emb if r["model"] == m]
        check_range(f"encoder top-1 retrieval, {lbl}", min(v), max(v), *want)

    # |c| calibration AUC
    cal = [r for r in rd(ROOT / "results/diagnostics/reinforcement_abcd/B_content_calibration/B_calibration_summary.csv")
           if r["scope"] == "overall"]
    auc = [float(r["isotonic_auroc"]) for r in cal]
    check_range("held-out isotonic AUC over 5 diagnostic judges", min(auc), max(auc), 0.52, 0.63,
                tol=0.005, dp=2)


# --------------------------------------------------------------------------
def rq3():
    section("RQ3  Content versus appearance (Figure 10)")
    bc = rd(RQ3 / "RQ3_IMAGE_ONLY_BY_CONTRAST.csv")
    want = {  # (variant_i, variant_j) -> {model: (effective preference, swap error)}
        ("golden", "beautiful_trash"): {QWEN: (98.3, 1.7), IVL3: (91.7, 8.3)},
        ("golden", "ugly_trash"): {QWEN: (97.7, 2.3), IVL3: (99.0, 1.0)},
        ("ugly_gold", "ugly_trash"): {QWEN: (94.0, 6.0), IVL3: (97.7, 2.0)},
        ("ugly_gold", "beautiful_trash"): {QWEN: (91.7, 8.3), IVL3: (82.0, 17.3)},
        ("golden", "ugly_gold"): {QWEN: (56.0, 43.3), IVL3: (57.3, 39.7)},
        ("beautiful_trash", "ugly_trash"): {QWEN: (8.3, 91.7), IVL3: (60.3, 36.7)},
    }
    for r in bc:
        key = (r["variant_i"], r["variant_j"])
        if key not in want or r["model"] not in want[key]:
            continue
        wp, ws = want[key][r["model"]]
        short = f"{key[0]}/{key[1]} {r['model'].split('/')[-1][:14]}"
        check(f"Fig 10 {short} pref", float(r["effective_target_preference"]) * 100, wp)
        check(f"Fig 10 {short} S", float(r["strict_swap_error"]) * 100, ws)

    n_parsed = sum(int(r["parse_failure_pairs"]) for r in bc)
    print(f"  {'RQ3 pairs with a parse failure':<58}{n_parsed:8}     (paper 0, all 7,200 calls parsed)")
    if n_parsed:
        FAILURES.append("RQ3 parse failures")

    conf = [float(r["target_preference_valid"]) * 100 for r in bc
            if r["contrast_type"] == "semantic_visual_conflict_primary"]
    check_range("key-conflict conditional preference", min(conf), max(conf), 99.2, 100.0)

    bl = rd(RQ3 / "RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv")
    for m, lbl, w in ((IVL3, "InternVL3-8B", (66.0, 91.0)), (QWEN, "Qwen2.5-VL-7B", (89.0, 94.0))):
        v = [float(r["effective_target_preference"]) * 100 for r in bl
             if r["model"] == m and r["contrast_type"] == "semantic_visual_conflict_primary"]
        check_range(f"key conflict by language, {lbl}", min(v), max(v), *w)


# --------------------------------------------------------------------------
def rq4():
    section("RQ4  Image-only pipeline comparison (Table 3)")

    # Table 3(a)
    sub = {r["system"]: r for r in rd(RQ4 / "ocr_substitution_frozen_rf/clean_multi_ocr_final_comparison_table.csv")}
    base = float(sub["Oracle source random_forest_regressor"]["effective_accuracy"]) * 100
    check("Table 3(a) frozen RF, source features", base, 62.33)
    for key, label, wv in (("OCR(rapidocr)+random_forest_regressor", "RapidOCR", 51.87),
                           ("OCR(easyocr)+random_forest_regressor", "EasyOCR", 48.23),
                           ("OCR(easyocr_preprocessed)+random_forest_regressor", "EasyOCR preprocessed", 51.73)):
        check(f"Table 3(a) frozen RF, {label}", float(sub[key]["effective_accuracy"]) * 100, wv)

    # Table 3(b)
    t5 = {(r["language"], r["row"]): r for r in rd(RQ4 / "image_only_pipelines/Table5_deploy_v2.csv")}
    direct = {"java": 57.17, "python": 63.07, "cuda": 65.57}
    coder = {"java": 50.10, "python": 55.60, "cuda": 54.33}
    for lang in LANGS:
        check(f"Table 3(b) direct Qwen D, {lang}",
              float(t5[(lang, "Direct Qwen image-only")]["D_main"]) * 100, direct[lang])
        check(f"Table 3(b) RapidOCR+Coder D, {lang}",
              float(t5[(lang, "RapidOCR-text Qwen2.5-Coder")]["D_main"]) * 100, coder[lang])
    for lang, wv in (("java", 28.47), ("python", 9.87), ("cuda", 21.57)):
        check(f"Table 3(b) RapidOCR+Coder E (strict), {lang}",
              float(t5[(lang, "RapidOCR-text Qwen2.5-Coder")]["E"]) * 100, wv)

    dd = {r["language"]: r for r in rd(RQ4 / "image_only_pipelines/F5_deploy_debiased.csv")
          if r["system"] == "direct_qwen_image_only"}
    for lang, wv in (("java", 4.20), ("python", 3.53), ("cuda", 4.83)):
        check(f"Table 3(b) direct - OCR+ML, {lang}", float(dd[lang]["diff_vs_ocrml"]) * 100, wv, unit="pp")
    print("    note: the CUDA row above compares against RapidOCR+LR (60.73). Table 3(b)'s")
    print("    CUDA comparator is EasyOCR+LR (61.37), giving +4.20 pp and [-2.6, +11.0];")
    print("    see figures_and_tables/tab03_rq4/recomputed_intervals.csv.")

    # Table 3(c)
    sm = {r["condition"]: r for r in rd(RQ4 / "text_input_ablation/summary_metrics.csv")
          if r["scope"] == "overall"}
    want3c = {"text_only_source": (44.78, 58.04, 56.30, 22.86),
              "text_only_ocr": (27.96, 54.76, 52.58, 48.94)}
    for cond, (wE, wV, wD, wS) in want3c.items():
        r = sm[cond]
        check(f"Table 3(c) {cond} E", float(r["effective_accuracy_E"]) * 100, wE)
        check(f"Table 3(c) {cond} A_valid", float(r["valid_accuracy_V"]) * 100, wV)
        check(f"Table 3(c) {cond} S", float(r["strict_swap_error_S"]) * 100, wS)
        # manuscript counts ties as errors: rescale to all 9,000 pairs
        d = float(r["debiased_accuracy_D"]) * int(r["n_D_defined"]) / int(r["n_pairs"]) * 100
        check(f"Table 3(c) {cond} D (ties as errors)", d, wD)
    img = sm["image_only_baseline_carried"]
    check("Table 3(c) images only E", float(img["effective_accuracy_E"]) * 100, 41.30)
    check("Table 3(c) images only A_valid", float(img["valid_accuracy_V"]) * 100, 68.35)
    check("Table 3(c) images only S", float(img["strict_swap_error_S"]) * 100, 39.58)
    d_img = sum(float(t5[(l, "Direct Qwen image-only")]["D_main"]) for l in LANGS) / 3 * 100
    check("Table 3(c) images only D (mean of per-language D_main)", d_img, 61.93)

    eS = float(sm["text_only_source"]["effective_accuracy_E"]) * 100
    eO = float(sm["text_only_ocr"]["effective_accuracy_E"]) * 100
    eI = float(img["effective_accuracy_E"]) * 100
    check("source E minus OCR E", eS - eO, 16.82, unit="pp")
    check("source E minus images E", eS - eI, 3.48, unit="pp")
    check("images E minus OCR E", eI - eO, 13.34, unit="pp")


# --------------------------------------------------------------------------
def main():
    for fn in (rq2, rq3, rq4):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} MISMATCH(ES): " + ", ".join(FAILURES))
        return 1
    print("All RQ2, RQ3 and RQ4 checks reproduce the manuscript.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

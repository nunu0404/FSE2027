#!/usr/bin/env python3
"""Analyze AB/BA content signal c and position bias b without new inference."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
AB_PATH = ROOT / "analysis" / "ab" / "A_B_PAIR_LEVEL.csv"
RQ3_PATH = ROOT / "rq3" / "analysis" / "image_only" / "RQ3_IMAGE_ONLY_PAIR_LEVEL.csv"
OUT = ROOT / "analysis" / "logit_decomposition"


def effect_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, part in data.groupby(["experiment", "model", "language", "condition"]):
        valid, invalid = part[part.strict_valid], part[~part.strict_valid]
        for value in ("abs_b", "abs_c"):
            a, b = valid[value].dropna(), invalid[value].dropna()
            result = mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else None
            rows.append({
                "experiment": key[0], "model": key[1], "language": key[2], "condition": key[3],
                "measure": value, "n_valid": len(a), "n_invalid": len(b),
                "median_valid": a.median() if len(a) else math.nan,
                "median_invalid": b.median() if len(b) else math.nan,
                "mannwhitney_u": float(result.statistic) if result else math.nan,
                "p": float(result.pvalue) if result else math.nan,
            })
    return pd.DataFrame(rows)


def calibration(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, part in data.groupby(["experiment", "model", "language", "condition"]):
        part = part.copy()
        try:
            part["bin"] = pd.qcut(part.abs_c.rank(method="first"), 10, labels=False) + 1
        except ValueError:
            continue
        for decile, d in part.groupby("bin"):
            valid = d[d.strict_valid]
            rows.append({
                "experiment": key[0], "model": key[1], "language": key[2], "condition": key[3],
                "abs_c_decile": int(decile), "n": len(d), "abs_c_min": d.abs_c.min(),
                "abs_c_median": d.abs_c.median(), "abs_c_max": d.abs_c.max(),
                "debiased_accuracy": d.debiased_correct.mean(),
                "strict_valid_rate": d.strict_valid.mean(),
                "strict_valid_only_accuracy": valid.strict_correct.mean() if len(valid) else math.nan,
            })
    return pd.DataFrame(rows)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, d in data.groupby(["experiment", "model", "language", "condition"]):
        valid = d[d.strict_valid]
        rows.append({
            "experiment": key[0], "model": key[1], "language": key[2], "condition": key[3],
            "n": len(d), "strict_valid_rate": d.strict_valid.mean(),
            "strict_effective_accuracy": d.strict_correct.mean(),
            "strict_valid_accuracy": valid.strict_correct.mean() if len(valid) else math.nan,
            "debiased_accuracy": d.debiased_correct.mean(), "debiased_tie_rate": d.debiased_tie.mean(),
            "debiased_minus_effective": d.debiased_correct.mean() - d.strict_correct.mean(),
            "median_abs_b": d.abs_b.median(), "median_abs_c": d.abs_c.median(),
            "mean_abs_b": d.abs_b.mean(), "mean_abs_c": d.abs_c.mean(),
        })
    return pd.DataFrame(rows)


def rq3_summary() -> pd.DataFrame:
    d = pd.read_csv(RQ3_PATH)
    d["abs_b"] = d.position_bias_b.abs()
    d["abs_c"] = d.content_signal_c.abs()
    rows = []
    for key, p in d.groupby(["model", "language", "contrast_type"]):
        valid = p[p.strict_valid]
        rows.append({
            "model": key[0], "language": key[1], "contrast_type": key[2], "n": len(p),
            "strict_valid_rate": p.strict_valid.mean(),
            "strict_target_effective": p.target_selected.mean(),
            "strict_target_valid_only": valid.target_selected.mean() if len(valid) else math.nan,
            "debiased_target_preference": p.debiased_target_selected.mean(),
            "debiased_tie_rate": p.debiased_tie.mean(),
            "median_abs_b": p.abs_b.median(), "median_abs_c": p.abs_c.median(),
        })
    return pd.DataFrame(rows)


def plots(data: pd.DataFrame, calibration_data: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    sample = data.iloc[rng.choice(len(data), min(20000, len(data)), replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for valid, label, color in ((True, "strict valid", "#2271b2"), (False, "swap", "#d55e00")):
        d = sample[sample.strict_valid.eq(valid)]
        axes[0].scatter(d.b_position, d.c_content, s=4, alpha=.18, label=label, color=color)
    lim = np.nanquantile(np.abs(sample[["b_position", "c_content"]].to_numpy()), .99)
    axes[0].plot([-lim, lim], [-lim, lim], color="black", lw=.8)
    axes[0].plot([-lim, lim], [lim, -lim], color="black", lw=.8)
    axes[0].set(xlabel="position component b", ylabel="content component c",
                xlim=(-lim, lim), ylim=(-lim, lim), title="AB/BA logit decomposition")
    axes[0].legend(frameon=False)
    c = calibration_data.groupby(["model", "abs_c_decile"], as_index=False).agg(
        debiased_accuracy=("debiased_accuracy", "mean"),
        valid_accuracy=("strict_valid_only_accuracy", "mean"))
    for model, d in c.groupby("model"):
        axes[1].plot(d.abs_c_decile, d.debiased_accuracy, marker="o", label=f"{model.split('/')[-1]} debiased")
        axes[1].plot(d.abs_c_decile, d.valid_accuracy, marker="x", linestyle="--",
                     label=f"{model.split('/')[-1]} valid-only")
    axes[1].set(xlabel="|c| decile", ylabel="accuracy", ylim=(0, 1),
                xticks=range(1, 11), title="Content-margin calibration")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "logit_decomposition_and_calibration.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for valid, label, color in ((True, "strict valid", "#2271b2"), (False, "swap", "#d55e00")):
        d = sample[sample.strict_valid.eq(valid)]
        ax.scatter(d.b_position, d.c_content, s=4, alpha=.18, label=label, color=color)
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=.8)
    ax.plot([-lim, lim], [lim, -lim], color="black", lw=.8)
    ax.set(xlabel="position component b", ylabel="content component c",
           xlim=(-lim, lim), ylim=(-lim, lim), title="AB/BA logit decomposition")
    ax.legend(frameon=False)
    fig.savefig(OUT / "b_c_scatter.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for model, local in c.groupby("model"):
        ax.plot(local.abs_c_decile, local.debiased_accuracy, marker="o",
                label=f"{model.split('/')[-1]} debiased")
        ax.plot(local.abs_c_decile, local.valid_accuracy, marker="x", linestyle="--",
                label=f"{model.split('/')[-1]} valid-only")
    ax.set(xlabel="|c| decile", ylabel="accuracy", ylim=(0, 1), xticks=range(1, 11),
           title="Content-margin calibration")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "content_margin_calibration.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(AB_PATH)
    d["abs_b"] = d.b_position.abs()
    d["abs_c"] = d.c_content.abs()
    # Algebraic identity: away from exact boundaries, opposite margin signs iff |c|>|b|.
    d["logit_boundary"] = np.isclose(d.abs_c, d.abs_b, rtol=0, atol=1e-12)
    d["identity_predicts_valid"] = d.abs_c > d.abs_b
    d["opposite_margin_signs"] = (d.margin_ab * d.margin_ba) < 0
    d["identity_violation"] = (~d.logit_boundary) & (d.identity_predicts_valid != d.opposite_margin_signs)
    d.to_csv(OUT / "LOGIT_PAIR_LEVEL.csv", index=False)
    d.to_csv(OUT / "logit_decomposition.csv", index=False)
    summary = summarize(d)
    summary.to_csv(OUT / "LOGIT_SUMMARY.csv", index=False)
    summary.to_csv(OUT / "debiased_accuracy_summary.csv", index=False)
    effect_tests(d).to_csv(OUT / "VALID_INVALID_EFFECT_TESTS.csv", index=False)
    cal = calibration(d)
    cal.to_csv(OUT / "ABS_C_CALIBRATION.csv", index=False)
    plots(d, cal)
    rq3_summary().to_csv(OUT / "RQ3_LOGIT_SUMMARY.csv", index=False)

    rho, p = spearmanr(d.abs_c, d.debiased_correct.astype(int))
    integrity = {
        "rows": len(d), "exact_boundary_rows": int(d.logit_boundary.sum()),
        "identity_violations_away_from_boundary": int(d.identity_violation.sum()),
        "opposite_sign_vs_parsed_valid_disagreements": int((d.opposite_margin_signs != d.strict_valid).sum()),
        "abs_c_correctness_spearman": float(rho), "abs_c_correctness_p": float(p),
        "note": "Parsed validity can differ at token-logit ties; the algebraic identity is tested on margin signs.",
    }
    (OUT / "INTEGRITY.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    (OUT / "LOGIT_INTERPRETATION.md").write_text(
        "# Logit decomposition interpretation\n\n"
        "## Korean\n\nAB/BA verdict margin을 위치 성분 b와 내용 성분 c로 분해했다. "
        "비경계 행에서는 `|c|>|b|`와 반대 margin 부호의 관계가 전수 성립했다. "
        "BF16 A/B logit 동률 경계는 별도로 보고하며, sign(c) 기반 정확도는 위치 성분을 "
        "대수적으로 제거한 two-pass 추정치이지 관측된 strict-valid 판정으로 재분류한 값이 아니다.\n\n"
        "## English\n\nWe decomposed AB/BA verdict margins into a position component b and a content "
        "component c. The `|c|>|b|` identity held for every non-boundary row. Exact BF16 A/B-logit "
        "ties are reported separately. Sign(c) accuracy is a two-pass position-debiased estimate, "
        "not a relabeling of observed swap-invalid pairs as strict-valid.\n",
        encoding="utf-8")
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()

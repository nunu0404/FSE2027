#!/usr/bin/env python3
"""Analyze the validated Qwen3 deploy-v2 direct-visual replication."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
RUN = OUT / "inference/deployment/qwen3_direct_visual"
DEPLOY = ROOT / "fse2027/external_runs/deploy_v2_logit_20260731"
PAIR_PATH = DEPLOY / "data/F5_FROZEN_PAIRS.csv"
OCR_PATH = DEPLOY / "data/F5_OCRML_PAIR_PREDICTIONS.csv"
OLD_RAW_PATH = DEPLOY / "inference/direct_qwen_image_only.jsonl"
LANGUAGES = ["java", "python", "cuda"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(OUT), "rev-parse", "HEAD"], text=True
    ).strip()


def chosen_snippet(row: pd.Series) -> str | None:
    if row.parsed_choice == "A":
        return row.snippet_first
    if row.parsed_choice == "B":
        return row.snippet_second
    return None


def build_pairs(calls: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    calls = calls.copy()
    calls["selected"] = calls.apply(chosen_snippet, axis=1)
    lookup = pairs.set_index("pair_id")
    rows = []
    for pair_id, group in calls.groupby("pair_id", sort=False):
        group = group.set_index("order")
        ab, ba = group.loc["AB"], group.loc["BA"]
        pair = lookup.loc[pair_id]
        valid = bool(
            pd.notna(ab.selected)
            and pd.notna(ba.selected)
            and ab.selected == ba.selected
        )
        strict_choice = ab.selected if valid else None
        margin_ab = float(ab.margin)
        margin_ba = float(ba.margin)
        c = (margin_ab - margin_ba) / 2.0
        b = (margin_ab + margin_ba) / 2.0
        d_choice = pair.snippet_i if c > 0 else pair.snippet_j if c < 0 else None
        rows.append(
            {
                "pair_id": pair_id,
                "language": pair.language,
                "difficulty": pair.difficulty,
                "snippet_i": pair.snippet_i,
                "snippet_j": pair.snippet_j,
                "human_preference": pair.human_preference,
                "valid": valid,
                "strict_choice": strict_choice,
                "strict_correct": valid and strict_choice == pair.human_preference,
                "margin_AB": margin_ab,
                "margin_BA": margin_ba,
                "c": c,
                "b": b,
                "tie_c": c == 0,
                "boundary": c != 0 and abs(c) == abs(b),
                "bc_rule_evaluable": c != 0 and abs(c) != abs(b),
                "bc_valid_prediction": abs(c) > abs(b),
                "bc_rule_violation": (
                    c != 0 and abs(c) != abs(b) and valid != (abs(c) > abs(b))
                ),
                "d_choice": d_choice,
                "d_correct": d_choice == pair.human_preference,
                "ab_correct": ab.selected == pair.human_preference,
                "ba_correct": ba.selected == pair.human_preference,
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(
    frame: pd.DataFrame, reps: int, seed: int
) -> tuple[float, float]:
    """Two-endpoint snippet bootstrap using dyadic multiplicity weights."""
    snippets = sorted(set(frame.snippet_i) | set(frame.snippet_j))
    lookup = {snippet: index for index, snippet in enumerate(snippets)}
    left = frame.snippet_i.map(lookup).to_numpy()
    right = frame.snippet_j.map(lookup).to_numpy()
    diff = (
        frame.d_correct.astype(float) - frame.ocr_correct.astype(float)
    ).to_numpy()
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(snippets), 1.0 / len(snippets))
    estimates = []
    for start in range(0, reps, 500):
        size = min(500, reps - start)
        counts = rng.multinomial(len(snippets), probabilities, size=size)
        weights = counts[:, left] * counts[:, right]
        denominator = weights.sum(axis=1)
        numerator = weights @ diff
        estimates.append(
            np.divide(
                numerator,
                denominator,
                out=np.full(size, np.nan),
                where=denominator > 0,
            )
        )
    values = np.concatenate(estimates)
    return tuple(np.nanquantile(values, [0.025, 0.975]))


def exact_mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    pvalue = 1.0 if discordant == 0 else binomtest(a_only, discordant, 0.5).pvalue
    return a_only, b_only, float(pvalue)


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy())
    raw = pvalues.to_numpy()[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, raw * (len(raw) - np.arange(len(raw))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=pvalues.index)


def old_run_consistency(calls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_json(OLD_RAW_PATH, lines=True)[
        ["pair_id", "language", "order", "parsed_choice"]
    ].rename(columns={"parsed_choice": "old_choice"})
    merged = calls.merge(
        old,
        on=["pair_id", "language", "order"],
        how="inner",
        validate="one_to_one",
    )
    merged["agreement"] = merged.parsed_choice.eq(merged.old_choice)
    merged["abs_margin"] = merged.margin.abs()
    rows = []
    for language, frame in merged.groupby("language", sort=True):
        disagree = frame[~frame.agreement]
        agree = frame[frame.agreement]
        rows.append(
            {
                "language": language,
                "new_calls": int((calls.language == language).sum()),
                "matched_calls": len(frame),
                "verdict_agreement": float(frame.agreement.mean()),
                "disagree_calls": len(disagree),
                "disagree_new_abs_margin_median": float(disagree.abs_margin.median()),
                "disagree_new_abs_margin_p90": float(disagree.abs_margin.quantile(0.9)),
                "disagree_new_abs_margin_max": float(disagree.abs_margin.max()),
                "agree_new_abs_margin_median": float(agree.abs_margin.median()),
            }
        )
    details = merged.loc[
        ~merged.agreement,
        [
            "language",
            "pair_id",
            "order",
            "parsed_choice",
            "old_choice",
            "margin",
            "abs_margin",
        ],
    ]
    return pd.DataFrame(rows), details


def main() -> None:
    validation = json.loads((RUN / "validation.json").read_text())
    if not validation.get("gate_pass"):
        raise RuntimeError("deployment inference validation did not pass")
    calls = pd.read_json(RUN / "raw.jsonl", lines=True)
    pairs = pd.read_csv(PAIR_PATH)
    expected_calls = 2 * len(pairs)
    if len(calls) != expected_calls:
        raise RuntimeError(f"expected {expected_calls} calls, found {len(calls)}")
    pair_frame = build_pairs(calls, pairs)
    if len(pair_frame) != len(pairs):
        raise RuntimeError("deployment pair reconstruction is incomplete")
    if int(pair_frame.bc_rule_violation.sum()) != 0:
        raise RuntimeError(
            f"b/c validity rule failed for {int(pair_frame.bc_rule_violation.sum())} pairs"
        )

    ocr = pd.read_csv(OCR_PATH)[
        ["pair_id", "language", "recomputed_correct", "selected_model"]
    ].rename(columns={"recomputed_correct": "ocr_correct"})
    pair_frame = pair_frame.merge(
        ocr, on=["pair_id", "language"], how="left", validate="one_to_one"
    )
    if pair_frame.ocr_correct.isna().any():
        raise RuntimeError("missing OCR+ML comparator prediction")
    pair_frame["ocr_correct"] = pair_frame.ocr_correct.astype(bool)

    rows = []
    for index, language in enumerate(LANGUAGES):
        frame = pair_frame[pair_frame.language.eq(language)].copy()
        n = len(frame)
        valid_n = int(frame.valid.sum())
        strict_correct = int(frame.strict_correct.sum())
        ties = int(frame.tie_c.sum())
        boundaries = int(frame.boundary.sum())
        d_correct = int(frame.d_correct.sum())
        ocr_e = float(frame.ocr_correct.mean())
        ci_lo, ci_hi = cluster_bootstrap_ci(frame, 10_000, 20260830 + index)
        d_only, ocr_only, pvalue = exact_mcnemar(
            frame.d_correct, frame.ocr_correct
        )
        rows.append(
            {
                "run_family": "latest_vlm_extension_20260830",
                "system": "Direct Qwen3-VL-8B image-only",
                "language": language,
                "n": n,
                "valid_pairs": valid_n,
                "E": strict_correct / n,
                "V": strict_correct / valid_n if valid_n else np.nan,
                "S": 1 - valid_n / n,
                "D_main": d_correct / n,
                "D_excl": float(frame.loc[~frame.tie_c, "d_correct"].mean()),
                "D_half": (d_correct + 0.5 * ties) / n,
                "AB_only": float(frame.ab_correct.mean()),
                "BA_only": float(frame.ba_correct.mean()),
                "first_position_choice_rate": float(
                    calls.loc[calls.language.eq(language), "parsed_choice"].eq("A").mean()
                ),
                "ties": ties,
                "boundaries": boundaries,
                "ocrml_E": ocr_e,
                "ocrml_model": frame.selected_model.iloc[0],
                "diff_vs_ocrml": d_correct / n - ocr_e,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "qwen_only_correct": d_only,
                "ocrml_only_correct": ocr_only,
                "mcnemar_p": pvalue,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["holm_p"] = holm_adjust(metrics.mcnemar_p)
    metrics["category"] = np.where(
        metrics.diff_vs_ocrml.le(0),
        "a",
        np.where(metrics.ci_lo.gt(0), "c", "b"),
    )

    analysis = OUT / "analysis/deployment"
    analysis.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(analysis / "deployment_pair_level.csv", index=False)
    metrics.to_csv(analysis / "deployment_metrics.csv", index=False)
    consistency, disagreements = old_run_consistency(calls)
    consistency.to_csv(analysis / "old_qwen25_consistency.csv", index=False)
    disagreements.to_csv(analysis / "old_qwen25_disagreement_calls.csv", index=False)

    strongest = int(metrics.category.map({"a": 0, "b": 1, "c": 2}).max())
    overall = {0: "(a)", 1: "(b)", 2: "(c)"}[strongest]
    lines = [
        "# Qwen3 deployment result",
        "",
        f"The preregistered overall decision is **{overall}**.",
        "",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"- {row.language}: D={row.D_main:.4f}, OCR+ML E={row.ocrml_E:.4f}, "
            f"difference={row.diff_vs_ocrml:+.4f}, 95% cluster CI "
            f"[{row.ci_lo:+.4f}, {row.ci_hi:+.4f}], exact McNemar "
            f"p={row.mcnemar_p:.4g}, Holm p={row.holm_p:.4g}, category ({row.category})."
        )
    lines += [
        "",
        "## Manuscript draft",
        "",
        "For Qwen3-VL-8B, we retained the A/B verdict logits in both presentation orders and defined the order-debiased decision as sign(c), where c=(m_AB-m_BA)/2. We compared that decision with the deterministic, supervised RapidOCR plus language-specific ML pipeline on the same frozen pairs. Confidence intervals use 10,000 two-endpoint snippet-cluster bootstrap replicates, and exact McNemar tests are Holm-adjusted across the three language-specific comparisons. The VLM result requires two calls per pair and verdict-logit access, whereas OCR+ML is supervised but deterministic at deployment time.",
    ]
    (analysis / "DEPLOYMENT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "package_git_commit": package_commit(),
        "inference_manifest": str((RUN / "manifest.json").relative_to(OUT)),
        "inference_validation_sha256": sha256(RUN / "validation.json"),
        "pair_manifest": str(PAIR_PATH.relative_to(ROOT)),
        "pair_manifest_sha256": sha256(PAIR_PATH),
        "pair_count_from_manifest": len(pairs),
        "expected_calls_formula": "2 * pair_count_from_manifest",
        "expected_calls": expected_calls,
        "ocrml_predictions_sha256": sha256(OCR_PATH),
        "old_qwen25_raw_sha256": sha256(OLD_RAW_PATH),
        "bootstrap": "10,000 two-endpoint snippet-cluster replicates",
        "mcnemar": "two-sided exact binomial McNemar",
        "multiplicity": "Holm across three language comparisons",
        "tie_rule": "c=0 incorrect for D_main; D_excl and D_half reported as sensitivity analyses",
        "decision": overall,
        "bc_rule": "strict_valid iff abs(c)>abs(b); zero empirical violations required",
        "bc_rule_violations": int(pair_frame.bc_rule_violation.sum()),
    }
    (analysis / "deployment_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

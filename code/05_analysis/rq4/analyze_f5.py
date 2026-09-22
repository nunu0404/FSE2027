#!/usr/bin/env python3
"""Analyze the preregistered F5 deployment-v2 logit run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path("/ANON/experiment_root")
RUN = Path("/ANON/scratch_rq1/deploy_v2_logit_20260731")
SYSTEMS = [
    "direct_qwen_image_only",
    "source_text_qwen_coder",
    "rapidocr_text_qwen_coder",
]
LANGUAGES = ["java", "python", "cuda"]
LABELS = {
    "direct_qwen_image_only": "Direct Qwen image-only",
    "source_text_qwen_coder": "Source-text Qwen2.5-Coder",
    "rapidocr_text_qwen_coder": "RapidOCR-text Qwen2.5-Coder",
}
OLD_OUTPUTS = {
    ("direct_qwen_image_only", "java"): ROOT / "experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
    ("direct_qwen_image_only", "python"): ROOT / "results/python_cuda_vlm_main_20260715/outputs/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
    ("direct_qwen_image_only", "cuda"): ROOT / "results/python_cuda_vlm_main_20260715/outputs/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
    ("source_text_qwen_coder", "java"): ROOT / "results/screenshot_only_ocr_clean_20260707/ocr_text_llm_raw/source_text_llm.jsonl",
    ("source_text_qwen_coder", "python"): ROOT / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/source_text_llm.jsonl",
    ("source_text_qwen_coder", "cuda"): ROOT / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/source_text_llm.jsonl",
    ("rapidocr_text_qwen_coder", "java"): ROOT / "results/screenshot_only_ocr_clean_20260707/ocr_text_llm_raw/ocr_text_llm_rapidocr.jsonl",
    ("rapidocr_text_qwen_coder", "python"): ROOT / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/ocr_text_llm_rapidocr.jsonl",
    ("rapidocr_text_qwen_coder", "cuda"): ROOT / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/ocr_text_llm_rapidocr.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_calls(system: str) -> pd.DataFrame:
    path = RUN / "inference" / f"{system}.jsonl"
    frame = pd.read_json(path, lines=True)
    frame["system"] = system
    return frame


def build_pairs(calls: pd.DataFrame) -> pd.DataFrame:
    calls = calls.copy()
    calls["selected"] = np.where(
        calls.parsed_choice.eq("A"),
        calls.snippet_first,
        np.where(calls.parsed_choice.eq("B"), calls.snippet_second, None),
    )
    identity = [
        "system", "language", "pair_id", "difficulty", "abs_z_diff",
        "snippet_i", "snippet_j", "human_preference",
    ]
    pivot = calls.pivot(index=identity, columns="order", values=["selected", "parsed_choice", "margin"]).reset_index()
    pivot.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in pivot.columns
    ]
    pivot["valid"] = (
        pivot.selected_AB.notna()
        & pivot.selected_BA.notna()
        & pivot.selected_AB.eq(pivot.selected_BA)
    )
    pivot["valid_correct"] = pivot.valid & pivot.selected_AB.eq(pivot.human_preference)
    pivot["c"] = (pivot.margin_AB - pivot.margin_BA) / 2.0
    pivot["b"] = (pivot.margin_AB + pivot.margin_BA) / 2.0
    pivot["tie_c"] = pivot.c.eq(0)
    pivot["boundary"] = pivot.c.abs().eq(pivot.b.abs()) & ~pivot.tie_c
    pivot["d_selected"] = np.where(
        pivot.c.gt(0), pivot.snippet_i, np.where(pivot.c.lt(0), pivot.snippet_j, None)
    )
    pivot["d_correct"] = pivot.d_selected.eq(pivot.human_preference)
    pivot["ab_correct"] = pivot.selected_AB.eq(pivot.human_preference)
    return pivot


def cluster_bootstrap_ci(frame: pd.DataFrame, reps: int, seed: int) -> tuple[float, float]:
    """Two-endpoint snippet bootstrap using dyadic multiplicity weights."""
    snippets = sorted(set(frame.snippet_i) | set(frame.snippet_j))
    lookup = {snippet: index for index, snippet in enumerate(snippets)}
    left = frame.snippet_i.map(lookup).to_numpy()
    right = frame.snippet_j.map(lookup).to_numpy()
    diff = (frame.d_correct.astype(float) - frame.ocr_correct.astype(float)).to_numpy()
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    chunk = 500
    probabilities = np.full(len(snippets), 1.0 / len(snippets))
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        counts = rng.multinomial(len(snippets), probabilities, size=size)
        weights = counts[:, left] * counts[:, right]
        denominator = weights.sum(axis=1)
        numerator = weights @ diff
        estimates.append(np.divide(numerator, denominator, out=np.full(size, np.nan), where=denominator > 0))
    values = np.concatenate(estimates)
    return tuple(np.nanquantile(values, [0.025, 0.975]))


def exact_mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else binomtest(a_only, discordant, 0.5).pvalue
    return a_only, b_only, float(p)


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy())
    raw = pvalues.to_numpy()[order]
    adjusted_sorted = np.maximum.accumulate(np.minimum(1.0, raw * (len(raw) - np.arange(len(raw)))))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=pvalues.index)


def old_run_comparison(calls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    details = []
    for system in SYSTEMS:
        for language in LANGUAGES:
            old = pd.read_json(OLD_OUTPUTS[(system, language)], lines=True)
            if "language" in old.columns:
                old = old[old.language.eq(language)]
            old = old[["pair_id", "parsed_ab", "parsed_ba"]].drop_duplicates("pair_id")
            old_calls = pd.concat(
                [
                    old[["pair_id", "parsed_ab"]].rename(columns={"parsed_ab": "old_choice"}).assign(order="AB"),
                    old[["pair_id", "parsed_ba"]].rename(columns={"parsed_ba": "old_choice"}).assign(order="BA"),
                ],
                ignore_index=True,
            )
            new = calls[(calls.system.eq(system)) & (calls.language.eq(language))]
            merged = new.merge(old_calls, on=["pair_id", "order"], how="inner", validate="one_to_one")
            merged["agreement"] = merged.parsed_choice.eq(merged.old_choice)
            merged["abs_margin"] = merged.margin.abs()
            summaries.append(
                {
                    "run_id": "deploy_v2_logit_20260731",
                    "system": system,
                    "language": language,
                    "n_new_calls": len(new),
                    "n_old_calls_matched": len(merged),
                    "n_agree": int(merged.agreement.sum()),
                    "verdict_agreement": float(merged.agreement.mean()),
                    "n_disagree": int((~merged.agreement).sum()),
                    "disagree_abs_margin_median": float(merged.loc[~merged.agreement, "abs_margin"].median()),
                    "disagree_abs_margin_p90": float(merged.loc[~merged.agreement, "abs_margin"].quantile(0.9)),
                    "disagree_abs_margin_max": float(merged.loc[~merged.agreement, "abs_margin"].max()),
                    "agree_abs_margin_median": float(merged.loc[merged.agreement, "abs_margin"].median()),
                    "old_asset": str(OLD_OUTPUTS[(system, language)].relative_to(ROOT)),
                    "old_asset_sha256": sha256(OLD_OUTPUTS[(system, language)]),
                }
            )
            details.append(
                merged.loc[~merged.agreement, ["system", "language", "pair_id", "order", "parsed_choice", "old_choice", "margin", "abs_margin"]]
            )
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def main() -> None:
    analysis = RUN / "analysis"
    analysis.mkdir(exist_ok=True)
    calls = pd.concat([read_calls(system) for system in SYSTEMS], ignore_index=True)

    duplicate_keys = int(calls.duplicated(["system", "language", "pair_id", "order"]).sum())
    missing_logits = int(calls[["logit_A", "logit_B", "margin"]].isna().any(axis=1).sum())
    parse_failures = int((~calls.parsed_choice.isin(["A", "B"])).sum())
    argmax_mismatch = int((~calls.argmax_matches_parsed.astype(bool)).sum())
    group_counts = calls.groupby(["system", "language", "order"]).size()
    if duplicate_keys or missing_logits or parse_failures or argmax_mismatch or not group_counts.eq(3000).all():
        raise RuntimeError("F5 integrity gate failed")

    pairs = build_pairs(calls)
    if len(pairs) != 27000:
        raise RuntimeError(f"Expected 27,000 system-pairs, found {len(pairs)}")
    ocr = pd.read_csv(RUN / "data/F5_OCRML_PAIR_PREDICTIONS.csv")
    ocr = ocr[["pair_id", "language", "recomputed_correct"]].rename(columns={"recomputed_correct": "ocr_correct"})
    pairs = pairs.merge(ocr, on=["pair_id", "language"], how="left", validate="many_to_one")
    pairs["ocr_correct"] = pairs.ocr_correct.astype(bool)
    pairs.to_csv(analysis / "F5_pair_level_metrics.csv", index=False)

    metric_rows = []
    for system_index, system in enumerate(SYSTEMS):
        for language_index, language in enumerate(LANGUAGES):
            frame = pairs[(pairs.system.eq(system)) & (pairs.language.eq(language))].copy()
            n = len(frame)
            n_valid = int(frame.valid.sum())
            n_valid_correct = int(frame.valid_correct.sum())
            ties = int(frame.tie_c.sum())
            boundaries = int(frame.boundary.sum())
            d_correct = int(frame.d_correct.sum())
            d_main = d_correct / n
            d_excl = frame.loc[~frame.tie_c, "d_correct"].mean()
            d_half = (d_correct + 0.5 * ties) / n
            ocr_e = frame.ocr_correct.mean()
            ci_lo, ci_hi = cluster_bootstrap_ci(
                frame, reps=10000, seed=20260731 + system_index * 100 + language_index
            )
            d_only, ocr_only, pvalue = exact_mcnemar(frame.d_correct, frame.ocr_correct)
            metric_rows.append(
                {
                    "run_id": "deploy_v2_logit_20260731",
                    "system": system,
                    "row": LABELS[system],
                    "language": language,
                    "n": n,
                    "n_valid": n_valid,
                    "n_valid_correct": n_valid_correct,
                    "E": n_valid_correct / n,
                    "V": n_valid_correct / n_valid if n_valid else np.nan,
                    "S": 1.0 - n_valid / n,
                    "D_main": d_main,
                    "D_excl": d_excl,
                    "D_half": d_half,
                    "AB_only": frame.ab_correct.mean(),
                    "ties": ties,
                    "boundaries": boundaries,
                    "ocrml_E": ocr_e,
                    "diff_vs_ocrml": d_main - ocr_e,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "d_only_correct": d_only,
                    "ocrml_only_correct": ocr_only,
                    "mcnemar_p": pvalue,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics["holm_p"] = holm_adjust(metrics.mcnemar_p)
    metrics["category"] = np.where(
        metrics.diff_vs_ocrml.le(0),
        "a",
        np.where(metrics.ci_lo.gt(0), "c", "b"),
    )
    metrics.to_csv(analysis / "F5_deploy_debiased.csv", index=False)

    table = metrics[["run_id", "language", "row", "n", "E", "V", "S", "D_main", "D_excl", "D_half", "AB_only", "ties", "boundaries"]].copy()
    gate = pd.read_csv(RUN / "data/F5_OCRML_GATE.csv")
    ocr_rows = pd.DataFrame(
        {
            "run_id": "deploy_v2_logit_20260731",
            "language": gate.language,
            "row": "RapidOCR + best ML (supervised)",
            "n": gate.n,
            "E": gate.effective_accuracy,
            "V": gate.effective_accuracy,
            "S": 0.0,
            "D_main": np.nan,
            "D_excl": np.nan,
            "D_half": np.nan,
            "AB_only": gate.effective_accuracy,
            "ties": 0,
            "boundaries": 0,
        }
    )
    pd.concat([table, ocr_rows], ignore_index=True).sort_values(["language", "row"]).to_csv(
        analysis / "Table5_deploy_v2.csv", index=False
    )

    consistency, disagreement = old_run_comparison(calls)
    consistency.to_csv(analysis / "F5_old_run_consistency.csv", index=False)
    disagreement.to_csv(analysis / "F5_old_run_disagreement_calls.csv", index=False)

    audit = {
        "run_id": "deploy_v2_logit_20260731",
        "calls": len(calls),
        "system_pairs": len(pairs),
        "duplicate_call_keys": duplicate_keys,
        "missing_logit_calls": missing_logits,
        "parse_failure_calls": parse_failures,
        "argmax_mismatch_calls": argmax_mismatch,
        "calls_per_system_language_order": {"|".join(key): int(value) for key, value in group_counts.items()},
        "bootstrap": "10,000 two-endpoint snippet-cluster replicates; shared multinomial snippet counts; dyad weight=w_i*w_j",
        "mcnemar": "two-sided exact binomial McNemar",
        "multiplicity": "Holm across 9 system-language comparisons",
    }
    (analysis / "F5_ANALYSIS_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    direct = metrics[metrics.system.eq("direct_qwen_image_only")]
    strongest = direct.category.map({"a": 0, "b": 1, "c": 2}).max()
    overall = {0: "(a)", 1: "(b)", 2: "(c)"}[int(strongest)]
    lines = [
        "# F5 deployment-v2 result",
        "",
        f"The preregistered overall decision is **{overall}**.",
        "",
        "## Direct Qwen versus the same-run RapidOCR+ML comparator",
        "",
    ]
    for row in direct.itertuples(index=False):
        lines.append(
            f"- {row.language}: D={row.D_main:.4f}, OCR+ML E={row.ocrml_E:.4f}, "
            f"difference={row.diff_vs_ocrml:+.4f}, 95% cluster CI [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}], "
            f"exact McNemar p={row.mcnemar_p:.4g}, Holm p={row.holm_p:.4g}, category ({row.category})."
        )
    lines += [
        "",
        "## English manuscript draft",
        "",
        "In the preregistered deployment-v2 rerun, we retained the A/B verdict logits for both presentation orders and computed the order-debiased decision as sign(c), where c=(m_AB-m_BA)/2. Comparisons against the deterministic RapidOCR plus supervised-ML pipeline use the same frozen pairs and are reported separately by language. The debiased VLM requires two model calls per pair and access to verdict logits, which in practice favors open or logprob-accessible models; the OCR+ML comparator is supervised whereas the VLM judge is zero-shot. All confidence intervals use 10,000 two-endpoint snippet-cluster bootstrap replicates, and exact McNemar tests are Holm-adjusted across the nine preregistered comparisons.",
    ]
    (analysis / "F5_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    inventory_rows = []
    for path in sorted(p for p in RUN.rglob("*") if p.is_file() and p.name != "SHA256_INVENTORY.csv"):
        inventory_rows.append(
            {"relative_path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    pd.DataFrame(inventory_rows).to_csv(RUN / "SHA256_INVENTORY.csv", index=False)


if __name__ == "__main__":
    main()

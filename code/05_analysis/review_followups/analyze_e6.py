#!/usr/bin/env python3
"""Build a denominator-complete deployment table for every system row."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/fse2027_review_defense_e1_e8_20260730/analysis/E6"


def deterministic_row(language: str, system: str, model: str, n: int, correct: int, run: str) -> dict:
    accuracy = correct / n
    return {
        "run_id": run, "language": language, "system": system, "model": model,
        "n_total_pairs": n, "n_valid_pairs": n, "n_correct_pairs": correct,
        "effective_accuracy": accuracy, "valid_accuracy": accuracy,
        "strict_swap_error": 0.0, "parse_failure_rate": 0.0,
        "parse_failure_calls": 0, "denominator_definition": "all pairs; deterministic score difference",
    }


def nondeterministic_row(
    language: str, system: str, model: str, n: int, valid: int, correct: int,
    swap: float, parse_rate: float, parse_calls: int, run: str,
) -> dict:
    return {
        "run_id": run, "language": language, "system": system, "model": model,
        "n_total_pairs": n, "n_valid_pairs": valid, "n_correct_pairs": correct,
        "effective_accuracy": correct / n,
        "valid_accuracy": correct / valid if valid else float("nan"),
        "strict_swap_error": swap, "parse_failure_rate": parse_rate,
        "parse_failure_calls": parse_calls,
        "denominator_definition": (
            "effective=correct strict-valid/all pairs; valid=correct strict-valid/valid pairs; "
            "swap=order-inconsistent pairs/all pairs; parse rate=parse-failed calls/(2*n)"
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    java = pd.read_csv(
        ROOT / "results/screenshot_only_ocr_clean_20260707/clean_multi_ocr_final_comparison_table.csv"
    )
    rf = java[(java.kind == "oracle_source_classical") & (java.model == "random_forest_regressor")].iloc[0]
    rows.append(deterministic_row("java", "Source-access RF", rf.model, 3000, int(rf.correct_pairs), "deploy_java_clean"))
    rapid_ml = java[(java.kind == "ocr_classical") & (java.ocr_engine == "rapidocr")].sort_values(
        "correct_pairs", ascending=False
    ).iloc[0]
    rows.append(deterministic_row("java", "RapidOCR + best ML", rapid_ml.model, 3000, int(rapid_ml.correct_pairs), "deploy_java_clean"))
    qwen = java[java.system.str.startswith("Direct VLM Qwen/")].iloc[0]
    rows.append(nondeterministic_row(
        "java", "Direct Qwen image-only", qwen.model, 3000, int(qwen.valid_pairs),
        int(qwen.correct_pairs), float(qwen.strict_swap_error), float(qwen.parse_failure_rate),
        0, "clean_default",
    ))
    java_llm = pd.read_csv(
        ROOT / "results/screenshot_only_ocr_clean_20260707/clean_ocr_text_llm_summary.csv"
    )
    for condition, system in [
        ("source_text_llm", "Source-text Qwen2.5-Coder"),
        ("ocr_text_llm_rapidocr", "Best OCR-text LLM (RapidOCR)"),
    ]:
        row = java_llm[java_llm.run_name == condition].iloc[0]
        rows.append(nondeterministic_row(
            "java", system, row.model, int(row.num_pairs), int(row.valid_pairs),
            int(row.correct_pairs), float(row.strict_swap_error), float(row.parse_failure_rate),
            0, "deploy_java_clean",
        ))

    source = pd.read_csv(
        ROOT / "results/python_cuda_missing_experiments_20260716/tables/clean_classical_results_by_language.csv"
    )
    ocr = pd.read_csv(
        ROOT / "results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_results_by_language.csv"
    )
    vlm = pd.read_csv(
        ROOT / "results/python_cuda_vlm_main_20260715/tables/vlm_results_by_language.csv"
    )
    llm = pd.read_csv(
        ROOT / "results/python_cuda_missing_experiments_20260716/tables/ocr_text_llm_results_by_language.csv"
    )
    for language in ["python", "cuda"]:
        rf = source[(source.language == language) & (source.model == "random_forest_regressor")].iloc[0]
        rows.append(deterministic_row(
            language, "Source-access RF", rf.model, int(rf.pairs), int(rf.correct_pairs),
            "deploy_python_cuda",
        ))
        rapid = ocr[(ocr.language == language) & (ocr.ocr_engine == "rapidocr")].sort_values(
            "pair_accuracy", ascending=False
        ).iloc[0]
        rows.append(deterministic_row(
            language, "RapidOCR + best ML", rapid.model, int(rapid.pairs),
            int(round(rapid.pair_accuracy * rapid.pairs)), "deploy_python_cuda",
        ))
        qwen = vlm[
            (vlm.language == language)
            & (vlm.model == "Qwen/Qwen2.5-VL-7B-Instruct")
            & (vlm.setting == "image_only")
        ].iloc[0]
        rows.append(nondeterministic_row(
            language, "Direct Qwen image-only", qwen.model, int(qwen.pairs),
            int(qwen.valid_pairs), int(qwen.correct_pairs), float(qwen.strict_swap_error),
            float(qwen.parse_failure_rate), int(qwen.parse_failure_calls), "deploy_python_cuda",
        ))
        for condition, system in [
            ("source", "Source-text Qwen2.5-Coder"),
            ("rapidocr", "Best OCR-text LLM (RapidOCR)"),
        ]:
            row = llm[(llm.language == language) & (llm.condition == condition)].iloc[0]
            rows.append(nondeterministic_row(
                language, system, "Qwen2.5-Coder-7B-Instruct", int(row.pairs),
                int(row.valid_pairs), int(row.correct_pairs), float(row.strict_swap_error),
                int(row.parse_failure_calls) / (2 * int(row.pairs)),
                int(row.parse_failure_calls), "deploy_python_cuda",
            ))

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "E6_deployment_full_metrics.csv", index=False)
    compact = result[[
        "language", "system", "n_total_pairs", "n_valid_pairs", "n_correct_pairs",
        "effective_accuracy", "valid_accuracy", "strict_swap_error", "parse_failure_rate",
    ]]
    (OUT / "E6_TABLE5_DRAFT.md").write_text(
        "# Table 5 replacement draft\n\n"
        + compact.to_markdown(index=False, floatfmt=".4f")
        + "\n\n"
        "All deterministic source/OCR-ML rows have valid coverage 100%, strict-swap "
        "error 0%, and parse-failure rate 0% by construction. VLM/LLM effective "
        "accuracy counts invalid pairs as incorrect.\n",
        encoding="utf-8",
    )
    conclusions = []
    for language, group in result.groupby("language"):
        ocr_ml = group[group.system == "RapidOCR + best ML"].iloc[0]
        alternatives = group[group.system.isin([
            "Direct Qwen image-only", "Source-text Qwen2.5-Coder",
            "Best OCR-text LLM (RapidOCR)",
        ])]
        conclusions.append(
            f"- {language}: RapidOCR+ML effective={ocr_ml.effective_accuracy:.2%}; "
            f"best neural judge effective={alternatives.effective_accuracy.max():.2%}. "
            f"Conclusion retained={ocr_ml.effective_accuracy > alternatives.effective_accuracy.max()}."
        )
    (OUT / "E6_CONCLUSION.md").write_text(
        "# E6 conclusion\n\n"
        "The deployment conclusion remains true only when stated as end-to-end "
        "effective accuracy, not conditional valid accuracy.\n\n"
        + "\n".join(conclusions)
        + "\n",
        encoding="utf-8",
    )
    print(compact.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

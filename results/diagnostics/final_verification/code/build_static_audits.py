#!/usr/bin/env python3
"""Build non-inference FSE 2027 verification artifacts from frozen assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/fse2027_final_verification"
DEPLOY = Path("/ANON/scratch_rq1/deploy_v2_logit_20260731")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_f1() -> None:
    source_path = ROOT / "results/rq1_model_battery_3lang_20260723/data/ml_predictions_9models_9000.csv"
    ocr_path = DEPLOY / "data/F5_OCRML_GATE.csv"
    source = pd.read_csv(source_path)
    rf = source[(source["language"] == "java") & (source["model"] == "random_forest_regressor")]
    ocr = pd.read_csv(ocr_path)
    svr = ocr[(ocr["language"] == "java") & (ocr["ocr_engine"] == "rapidocr") & (ocr["model"] == "svr")]
    if len(rf) != 3000 or len(svr) != 1:
        raise RuntimeError(f"Unexpected F1 rows: RF={len(rf)}, OCR={len(svr)}")
    rf_correct = int(rf["is_correct"].astype(bool).sum())
    rf_total = len(rf)
    svr_row = svr.iloc[0]
    svr_correct = int(svr_row["correct"])
    svr_total = int(svr_row["n"])
    rf_acc = rf_correct / rf_total
    svr_acc = svr_correct / svr_total
    rows = [
        {
            "record": "source_access_rf_java",
            "pipeline": "random_forest_regressor",
            "language": "java",
            "correct": rf_correct,
            "total": rf_total,
            "effective_accuracy": rf_acc,
            "display_percent_2dp": round(100 * rf_acc, 2),
            "source_file": str(source_path),
            "fields": "language,model,is_correct",
        },
        {
            "record": "rapidocr_ml_java",
            "pipeline": "RapidOCR+SVR",
            "language": "java",
            "correct": svr_correct,
            "total": svr_total,
            "effective_accuracy": svr_acc,
            "display_percent_2dp": round(100 * svr_acc, 2),
            "source_file": str(ocr_path),
            "fields": "language,ocr_engine,model,correct,n,effective_accuracy",
        },
        {
            "record": "ocr_loss_source_minus_ocrml",
            "pipeline": "RF minus RapidOCR+SVR",
            "language": "java",
            "correct": rf_correct - svr_correct,
            "total": rf_total,
            "effective_accuracy": rf_acc - svr_acc,
            "display_percent_2dp": round(100 * (rf_acc - svr_acc), 2),
            "source_file": f"{source_path};{ocr_path}",
            "fields": "difference of unrounded effective_accuracy",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "ocr_loss_audit.csv", index=False)


def build_f2() -> None:
    run = ROOT / "results/nvidia_qwen35_extension_pilot_20260713"
    manifest_path = run / "manifest.json"
    calls_path = run / "calls.jsonl"
    pairs_path = run / "pilot_pairs.csv"
    summary_path = run / "summary.csv"
    stats_path = run / "run_stats.json"
    manifest = json.loads(manifest_path.read_text())
    stats = json.loads(stats_path.read_text())
    calls = pd.read_json(calls_path, lines=True).drop_duplicates("call_key", keep="last")
    pairs = pd.read_csv(pairs_path)
    summary = pd.read_csv(summary_path)
    image_calls = calls[calls["condition"] == "image_only"]
    image_pairs = summary[summary["condition"] == "image_only"].iloc[0]
    pair_counts = (
        pairs.groupby(["dataset_name", "language", "difficulty"]).size().rename("pairs").reset_index()
    )
    metadata = {
        "audit_status": "partially_reconstructable_hosted_api_run",
        "exact_model_id": manifest["model"],
        "recommended_table_name": "Qwen3.5-122B-A10B (MoE, 10B active; non-thinking, NVIDIA API)",
        "architecture": {
            "type": "Mixture-of-Experts multimodal causal language model with vision encoder",
            "total_parameters": "122B per Qwen model card; NVIDIA page also displays a rounded/inconsistent 125B parameter label",
            "activated_parameters": "10B",
            "official_sources": [
                "https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
                "https://build.nvidia.com/qwen/qwen3.5-122b-a10b",
            ],
        },
        "execution": {
            "endpoint": manifest["endpoint"],
            "provider": "NVIDIA hosted NIM/API",
            "first_successful_call_utc": str(calls.loc[calls["status"] == "success", "finished_at_utc"].min()),
            "last_successful_call_utc": str(calls.loc[calls["status"] == "success", "finished_at_utc"].max()),
            "manifest_created_utc": manifest["created_at_utc"],
            "checkpoint_revision": None,
            "checkpoint_revision_status": "not recorded by hosted endpoint",
            "server_inference_engine": None,
            "server_inference_engine_version": None,
            "server_transformers_version": None,
            "dtype": None,
            "quantization": None,
            "server_chat_template": None,
            "unavailable_fields_reason": "The NVIDIA hosted API response and local manifest do not expose backend checkpoint revision, engine/library versions, dtype, quantization, or the rendered server chat template.",
        },
        "decoding": {
            "thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": int(manifest["temperature"]),
            "max_tokens": int(manifest["max_tokens"]),
            "seed": int(manifest["sample_seed"]),
            "stream": False,
            "top_p": None,
            "top_p_status": "not sent; provider default not recorded",
        },
        "prompt": {
            "system_prompt": None,
            "message_roles": ["user"],
            "template_source": str(ROOT / "experiments/rq0_viability/scripts/run_nvidia_qwen_extension_pilot.py"),
            "template_function": "prompt_for",
            "description": "Prompt B-like readability rubric with language substituted and exact FINAL_VERDICT: A/B output instruction",
        },
        "images": {
            "pair_file": str(pairs_path),
            "upstream_pair_file": manifest["pair_source"],
            "render_directory": manifest["render_dir"],
            "preprocessing": "No client-side resize/crop/normalization. Existing PNG bytes were sent as base64 data URLs when all files were <=180,000 bytes; otherwise uploaded as NVIDIA assets.",
        },
        "sample": {
            "total_pairs": int(len(pairs)),
            "claim_300_java_pairs": False,
            "explanation": "The 300-pair sample contains 150 Java, 75 Python, and 75 CUDA pairs; it is not 300 Java pairs.",
            "counts_by_dataset_language_difficulty": pair_counts.to_dict("records"),
            "conditions": manifest["conditions"],
            "orders": manifest["strict_swap_orders"],
            "successful_final_calls": int((calls["status"] == "success").sum()),
            "required_calls": int(stats["required_calls"]),
        },
        "image_only_raw_metrics": {
            "pairs": int(image_pairs["pairs"]),
            "valid_pairs": int(image_pairs["valid_pairs"]),
            "correct_pairs": int(image_pairs["correct_pairs"]),
            "effective_accuracy": float(image_pairs["effective_accuracy"]),
            "strict_swap_error": float(image_pairs["strict_swap_error"]),
            "parse_failed_calls": int(image_calls["parsed_choice"].isna().sum()),
            "parse_failure_denominator_calls": int(len(image_calls)),
            "parse_failure_rate": float(image_calls["parsed_choice"].isna().mean()),
            "paper_23_percent_reproduced": bool(image_pairs["effective_accuracy"] == 0.23),
            "paper_65_percent_reproduced": bool(image_pairs["strict_swap_error"] == 0.65),
            "paper_11_percent_exactly_reproduced": False,
            "parse_rate_note": "Image-only is 65/600 = 10.8333%, not exactly 11.00%. Across both modalities it is 131/1200 = 10.9167%, which rounds to 10.92% (or 10.9% at one decimal), not 11.00% at two decimals.",
        },
        "source_files": {p.name: {"absolute_path": str(p), "sha256": sha256(p)} for p in [manifest_path, calls_path, pairs_path, summary_path, stats_path]},
    }
    (OUT / "qwen35_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_f4() -> None:
    grid_path = ROOT / "results/grounded_protocol_3lang_20260721/analysis/ab/A_GRID_MCNEMAR.csv"
    pert_path = ROOT / "results/grounded_protocol_3lang_20260721/analysis/ab/B_PERTURBATION_MCNEMAR.csv"
    grid = pd.read_csv(grid_path)
    pert = pd.read_csv(pert_path)
    sig = grid[grid["holm_reject_0_05"].astype(bool)].copy()
    conditions = sorted(set(grid["condition"]) | set(grid["baseline"]))
    result = {
        "rendering_conditions_count": len(conditions),
        "rendering_conditions": conditions,
        "factors": {"display_mode": ["friendly_light", "mono_light", "monokai_dark"], "font_size_px": [20, 24], "wrap_columns": [60, 80], "line_numbers": [True]},
        "condition_name_note": "The lnon suffix means line numbers ON in this run.",
        "baseline": str(grid["baseline"].unique().item()),
        "section_5": {
            "definition": "11 non-baseline conditions compared with the baseline in each of 2 models x 3 languages",
            "formula": "11 x 6 = 66",
            "rows": int(len(grid)),
            "significant_rows": int(len(sig)),
            "holm_family": "11 tests separately within each experiment/model/language block; not one 66-test family",
            "significant_contrasts": sig.to_dict("records"),
            "source": str(grid_path),
        },
        "section_6_localization": {
            "definition": "All unordered pairs among 12 conditions within each model-language combination",
            "formula": "C(12,2) = 66 per model-language; 66 x 6 = 396 condition pairs",
            "metrics": ["fixed_AB", "fixed_BA", "both_valid"],
            "total_metric_rows": 1188,
            "holm_family": "66 tests separately within model/language/metric",
            "source_code": str(ROOT / "results/fse2027_reinforcement_abcd_20260729/code/run_analyses_abcd.py"),
        },
        "followup_96": {
            "definition": "Grounded RQ2 baseline contrasts across grid and perturbation batteries",
            "formula": "66 grid rows + 30 perturbation rows = 96",
            "grid_rows": int(len(grid)),
            "perturbation_rows": int(len(pert)),
            "perturbation_formula": "5 non-baseline perturbations x 6 model-language combinations = 30",
            "holm_scope": "Grid: 11 within each model-language; perturbation: 5 within each model-language",
            "source": str(pert_path),
        },
        "paper_ready_text": "The 66 RQ2 rows comprise 11 baseline comparisons in each of six model-language combinations, with Holm adjustment applied separately to each 11-test model-language family. The localization analysis is distinct: it evaluates all C(12,2)=66 unordered rendering-condition pairs within each model-language combination (396 condition pairs overall).",
    }
    (OUT / "rendering_contrast_definition.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_f1()
    build_f2()
    build_f4()


if __name__ == "__main__":
    main()

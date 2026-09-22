#!/usr/bin/env python3
"""Generate the final latest-VLM extension report from validated artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
MODEL_ORDER = ["qwen3", "internvl3_5", "gemma4"]
MODEL_NAMES = {
    "qwen3": "Qwen3-VL-8B-Instruct",
    "internvl3_5": "InternVL3.5-8B",
    "gemma4": "Gemma-4-12B-it",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value) -> str:
    return "-" if pd.isna(value) else f"{100 * float(value):.2f}"


def rho(value) -> str:
    return "-" if pd.isna(value) else f"{float(value):.3f}"


def require_validations() -> None:
    paths = [
        OUT / f"inference/full/{model}/validation.json" for model in MODEL_ORDER
    ] + [
        OUT / f"inference/robustness/{model}/validation.json"
        for model in ("qwen3", "internvl3_5")
    ] + [OUT / "inference/deployment/qwen3_direct_visual/validation.json"]
    for path in paths:
        if not path.is_file() or not json.loads(path.read_text()).get("gate_pass"):
            raise RuntimeError(f"validated artifact required: {path}")


def table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def main() -> None:
    require_validations()
    analysis = OUT / "analysis"
    primary = pd.read_csv(analysis / "primary_metrics.csv")
    generation = pd.read_csv(analysis / "generation_metric_comparison.csv")
    robust = pd.read_csv(analysis / "robustness_metrics.csv")
    robust_pairs = pd.read_csv(analysis / "robustness_paired_comparisons.csv")
    repeat = pd.read_csv(analysis / "robustness_primary_repeat_consistency.csv")
    deployment = pd.read_csv(analysis / "deployment/deployment_metrics.csv")
    deploy_consistency = pd.read_csv(
        analysis / "deployment/old_qwen25_consistency.csv"
    )
    protocol = json.loads((OUT / "config/protocol.json").read_text())
    sanity = json.loads((OUT / "audit/SANITY_CHECK.json").read_text())
    asset_audit = json.loads((OUT / "audit/PRE_INFERENCE_AUDIT.json").read_text())

    lines = [
        "# Latest VLM Extension Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope and protocol",
        "",
        "This isolated extension evaluates Qwen3-VL-8B-Instruct, InternVL3.5-8B, and Gemma-4-12B-it under the frozen three-language image-only protocol. Historical full inference and full metric reproduction were explicitly omitted; only the preregistered stored-output sanity subset was checked before new inference.",
        "",
        f"- Primary manifest: `{protocol['primary']['pair_manifest']}` ({protocol['primary']['pairs']:,} pairs; {protocol['primary']['calls_per_model']:,} AB/BA calls per model).",
        f"- Prompt SHA-256: `{asset_audit['primary']['prompt_sha256']}`.",
        f"- Manifest SHA-256: `{asset_audit['primary']['pair_manifest_sha256']}`.",
        f"- Image audit: {asset_audit['primary']['unique_pngs']} unique PNGs, {asset_audit['primary']['image_hash_mismatches']} hash mismatches.",
        "- Input: two separate rendered PNGs, image-only, no OCR/source text/system prompt/few-shot/reasoning.",
        "- Generation: BF16, no quantization, batch size 1, greedy `do_sample=False`, max 24 new tokens, seed 42.",
        "- Each process was isolated to exactly one physical GPU by `CUDA_VISIBLE_DEVICES=0` or `1`.",
        "",
        "## Environment and model revisions",
        "",
    ]
    env_rows = []
    for model in MODEL_ORDER:
        env = json.loads((OUT / f"environments/{model}_full.json").read_text())
        model_cfg = protocol["models"][model]
        env_rows.append(
            [
                MODEL_NAMES[model],
                f"`{model_cfg['revision']}`",
                env["python"],
                env["torch"],
                env["transformers"],
                f"GPU {env['gpu']['physical_index']}",
            ]
        )
    lines += table(
        ["Model", "Revision", "Python", "PyTorch", "Transformers", "GPU"],
        env_rows,
    )
    lines += [
        "",
        "## Pre-inference sanity gate",
        "",
        f"The fixed 18-pair stored-output sanity check passed: `{sanity['gate_pass']}`. It checked {sum(model['calls_checked'] for model in sanity['models'])} stored calls across predecessor Qwen and InternVL outputs. Parser mismatch, candidate-mapping error, missing-logit, margin-orientation error, argmax mismatch, and non-tie/non-boundary b/c-rule violation counts were all zero.",
        "",
        "## Primary results",
        "",
        "V is accuracy conditional on strict-swap validity; E counts invalid pairs as failures; S is strict-swap error. D is the two-order logit-averaged `sign(c)` accuracy with exact `c=0` ties counted as incorrect. Spearman is valid-only and uses the signed human rating gap.",
        "",
    ]
    primary_rows = []
    selection = primary[
        (primary.scope.eq("pooled")) | (primary.scope.eq("language"))
    ].copy()
    for model in MODEL_ORDER:
        for row in selection[selection.model_key.eq(model)].itertuples(index=False):
            primary_rows.append(
                [
                    MODEL_NAMES[model],
                    row.scope_value,
                    row.pairs,
                    row.valid_pairs,
                    pct(row.valid_accuracy),
                    pct(row.effective_accuracy),
                    pct(row.strict_swap_error),
                    pct(row.two_order_averaged_accuracy),
                    pct(row.first_position_choice_rate),
                    rho(row.spearman_valid_only),
                    row.content_ties,
                    row.boundaries,
                ]
            )
    lines += table(
        [
            "Model",
            "Scope",
            "N",
            "Valid N",
            "V %",
            "E %",
            "S %",
            "D %",
            "First %",
            "rho",
            "Ties",
            "Boundary",
        ],
        primary_rows,
    )
    lines += [
        "",
        "The complete language-by-difficulty table is in `analysis/primary_metrics.csv`.",
        "",
        "## Generation comparison",
        "",
    ]
    generation_rows = []
    pooled_generation = generation[generation.scope.eq("pooled")]
    for row in pooled_generation.itertuples(index=False):
        generation_rows.append(
            [
                f"{row.predecessor} -> {MODEL_NAMES[row.new_model]}",
                f"{100 * row.delta_valid_accuracy:+.2f}",
                f"{100 * row.delta_effective_accuracy:+.2f}",
                f"{100 * row.delta_strict_swap_error:+.2f}",
                f"{100 * row.delta_two_order_averaged_accuracy:+.2f}",
                f"{100 * row.delta_first_position_choice_rate:+.2f}",
                f"{row.delta_spearman_valid_only:+.3f}",
            ]
        )
    lines += table(
        ["Generation", "Delta V pp", "Delta E pp", "Delta S pp", "Delta D pp", "Delta first pp", "Delta rho"],
        generation_rows,
    )
    lines += [
        "",
        "Language- and difficulty-specific predecessor comparisons are in `analysis/generation_metric_comparison.csv`; pair-matched correct/incorrect transitions are in `analysis/generation_comparison.csv`.",
        "",
        "## Presentation and modality robustness",
        "",
    ]
    robust_rows = []
    robust_selection = robust[
        (robust.scope.eq("pooled")) | (robust.scope.eq("language"))
    ]
    for model in ("qwen3", "internvl3_5"):
        for row in robust_selection[robust_selection.model_key.eq(model)].itertuples(index=False):
            robust_rows.append(
                [
                    MODEL_NAMES[model],
                    row.condition,
                    row.scope_value,
                    pct(row.valid_accuracy),
                    pct(row.effective_accuracy),
                    pct(row.strict_swap_error),
                    pct(row.two_order_averaged_accuracy),
                    pct(row.first_position_choice_rate),
                    rho(row.spearman_valid_only),
                    row.parse_failure_calls,
                ]
            )
    lines += table(
        ["Model", "Condition", "Scope", "V %", "E %", "S %", "D %", "First %", "rho", "Parse failures"],
        robust_rows,
    )
    lines += [
        "",
        "All conditions use the same 900 pairs and both orders. Exact paired comparisons against separate-image image-only are in `analysis/robustness_paired_comparisons.csv`; exact primary-repeat agreement is in `analysis/robustness_primary_repeat_consistency.csv`.",
        "",
        "## Screenshot-only deployment replication",
        "",
    ]
    deploy_rows = []
    for row in deployment.itertuples(index=False):
        deploy_rows.append(
            [
                row.language,
                row.n,
                pct(row.V),
                pct(row.E),
                pct(row.S),
                pct(row.D_main),
                pct(row.ocrml_E),
                f"{100 * row.diff_vs_ocrml:+.2f}",
                f"[{100 * row.ci_lo:+.2f}, {100 * row.ci_hi:+.2f}]",
                f"{row.holm_p:.4g}",
                row.category,
            ]
        )
    lines += table(
        ["Language", "N", "V %", "E %", "S %", "Qwen3 D %", "OCR+ML E %", "Delta pp", "95% cluster CI pp", "Holm p", "Decision"],
        deploy_rows,
    )
    lines += [
        "",
        "The deployment comparison uses 10,000 two-endpoint snippet-cluster bootstrap replicates and two-sided exact McNemar tests with Holm correction across the three languages. Qwen3 D requires two calls and verdict-logit access; OCR+ML is supervised but deterministic.",
        "",
        "## Anomalies and integrity",
        "",
    ]
    anomaly_rows = []
    for model in MODEL_ORDER:
        validation = json.loads(
            (OUT / f"inference/full/{model}/validation.json").read_text()
        )
        anomaly_rows.append(
            [
                MODEL_NAMES[model],
                validation["actual_calls"],
                validation["duplicate_calls"],
                validation["parse_failures"],
                validation["missing_logits"],
                validation["argmax_mismatches"],
                validation["mapping_errors"],
            ]
        )
    lines += table(
        ["Model", "Calls", "Duplicates", "Parse failures", "Missing logits", "Argmax mismatch", "Mapping errors"],
        anomaly_rows,
    )

    pooled = primary[primary.scope.eq("pooled")].set_index("model_key")
    best_v = pooled.valid_accuracy.idxmax()
    best_e = pooled.effective_accuracy.idxmax()
    instability = (pooled.strict_swap_error > 0.10).any()
    d_gain = pooled.two_order_averaged_accuracy - pooled.effective_accuracy
    modality_effect = robust_pairs.delta_effective_accuracy.abs().max()
    deploy_point_advantage = (deployment.diff_vs_ocrml > 0).to_dict()
    lines += [
        "",
        "## Answers to the study questions",
        "",
        f"1. Substantial order instability in at least one newer model: **{'yes' if instability else 'no'}**; pooled strict-swap errors range from {pct(pooled.strict_swap_error.min())}% to {pct(pooled.strict_swap_error.max())}%.",
        "2. Persistence of the older-model phenomenon is determined by the model-specific generation comparison above; no cross-run values were merged.",
        f"3. Best pooled conditional valid accuracy: **{MODEL_NAMES[best_v]} ({pct(pooled.loc[best_v, 'valid_accuracy'])}%)**.",
        f"4. Best pooled effective accuracy: **{MODEL_NAMES[best_e]} ({pct(pooled.loc[best_e, 'effective_accuracy'])}%)**.",
        f"5. Two-order averaging changes effective accuracy by {100 * d_gain.min():+.2f} to {100 * d_gain.max():+.2f} percentage points across the new models.",
        f"6. The largest paired effective-accuracy shift across packaging/modality conditions is {100 * modality_effect:.2f} percentage points; cell-level direction and tests are in the paired-comparison CSV.",
        "7. Qwen3 deployment point-estimate advantages are reported separately by language in the table above; statistical decisions use the preregistered (a)/(b)/(c) rule.",
        "8. Any weakening or strengthening of the prior conclusions must be interpreted from the full model- and language-specific tables, including unfavorable cells.",
        "",
        "## Reproducibility inventory",
        "",
        f"Package Git commit at report generation: `{subprocess.check_output(['git', '-C', str(OUT), 'rev-parse', 'HEAD'], text=True).strip()}`.",
        "Every inference manifest records model/processor revision, runner SHA-256, package commit, prompt and pair-manifest hashes, generation settings, environment snapshot, and the single visible physical GPU. The machine-readable per-call, per-pair, metric, comparison, and validation artifacts remain under this result directory.",
    ]
    (OUT / "LATEST_VLM_EXTENSION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    inventory_rows = []
    for path in sorted(
        p for p in OUT.rglob("*") if p.is_file() and ".git" not in p.parts
    ):
        if path.name == "SHA256_INVENTORY.csv":
            continue
        inventory_rows.append(
            {
                "relative_path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(inventory_rows).to_csv(OUT / "SHA256_INVENTORY.csv", index=False)
    print(OUT / "LATEST_VLM_EXTENSION_REPORT.md")
    print(f"inventory_files={len(inventory_rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the human-readable F1-F4 report and non-circular SHA inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/fse2027_followup_f1_f4_20260731"


def pct(value: float) -> str:
    return "NA" if pd.isna(value) else f"{100 * value:.2f}%"


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def build_report() -> None:
    f1 = pd.read_csv(OUT / "analysis/F1/F1_deploy_debiased.csv")
    f4 = pd.read_csv(OUT / "analysis/F4/F4_bothvalid_floor.csv")
    f3 = pd.read_csv(OUT / "analysis/F3/F3_mismatch_call_summary.csv")
    f2 = pd.read_csv(OUT / "analysis/F2/F2_noise_ceiling.csv")
    f2j = pd.read_csv(OUT / "analysis/F2/F2_java_within.csv")
    refs = pd.read_csv(OUT / "analysis/F2/F2_table1_reference.csv")

    f1_rows = [
        [
            row.language,
            row.row,
            pct(row.E),
            pct(row.V),
            pct(row.S),
            "not measured",
            pct(row.ocrml_E),
        ]
        for row in f1.itertuples(index=False)
    ]
    f4_rows = [
        [
            "Qwen" if row.model.startswith("Qwen") else "InternVL",
            row.language,
            f"{row.n_mismatch_e4a}/{row.n_denominator_e4a}",
            pct(row.floor_e4a_cross_environment),
            f"{row.n_mismatch_e4a_vs_e4b}/{row.n_denominator_e4a_vs_e4b}",
            pct(row.floor_e4a_vs_e4b),
            f"{pct(row.flip_bothvalid_min)}--{pct(row.flip_bothvalid_max)}",
        ]
        for row in f4.itertuples(index=False)
    ]
    f3_rows = [
        [
            "Qwen" if row.model.startswith("Qwen") else "InternVL",
            row.language,
            f"{row.mismatch_calls}/{row.n_calls}",
            pct(row.mismatch_rate),
            f"{row.mismatch_abs_margin_median:.3f}",
            f"{row.match_abs_margin_median:.3f}",
            f"{row.mismatch_abs_c_median:.3f}",
            f"{row.mismatch_abs_b_median:.3f}",
            f"{row.mismatch_max_logit_diff_median:.3f}",
            f"{row.mismatch_max_logit_diff_max:.3f}",
            f"{row.mismatch_tie_c_calls}/{row.mismatch_boundary_calls}",
        ]
        for row in f3.itertuples(index=False)
    ]
    f2_rows = [
        [
            row.language,
            row.difficulty,
            row.estimator,
            pct(row.agreement_probability),
            f"[{pct(row.ci_lo)}, {pct(row.ci_hi)}]",
            row.n_pairs,
            pct(row.mean_tie_probability),
        ]
        for row in f2.itertuples(index=False)
    ]
    f2j_rows = [
        [
            row.benchmark,
            row.difficulty,
            row.estimator,
            pct(row.agreement_probability),
            f"[{pct(row.ci_lo)}, {pct(row.ci_hi)}]",
            f"{row.n_pairs}/{row.n_pairs_requested}",
            row.n_pairs_unavailable,
        ]
        for row in f2j.itertuples(index=False)
    ]
    max_refs = (
        refs.groupby(["language", "benchmark", "difficulty"], as_index=False)
        .agg(
            best_model_valid=("valid_accuracy", "max"),
            panel_ceiling=("ceiling_panel", "first"),
        )
        .sort_values(["language", "benchmark", "difficulty"])
    )
    ref_rows = [
        [
            row.language,
            row.benchmark,
            row.difficulty,
            pct(row.best_model_valid),
            pct(row.panel_ceiling),
            pct(row.best_model_valid / row.panel_ceiling),
        ]
        for row in max_refs.itertuples(index=False)
    ]

    report = f"""# FSE 2027 follow-up F1-F4 final report

Run package: `fse2027_followup_f1_f4_20260731`

New model-inference calls: **0**.

## Executive corrections

1. F1 cannot be computed from the deployment run because its AB/BA assets do
   not store verdict-token logits. D is reported as not measured in all nine
   neural row-language cells; no run was merged.
2. E4-A was mislabelled as a same-environment repeat. The original grid used
   Python 3.13.11 and E4-A used Python 3.10.12. Its disagreement is therefore
   cross-environment, not a same-environment floor.
3. Direct reaggregation of all 66 grid rendering cells gives a both-valid flip
   range of 0.00%--8.31%, not the manuscript's 0.6%--7.1%. The detailed
   condition-level table is authoritative until the manuscript scope is
   reconciled.
4. The public Dorn Java rater matrix reproduces 89/90 selected proxy means.
   Snippet `rq0_0099` has no reproducible rating column, so its nine
   within-Dorn pairs are explicitly unavailable.

## F1 deployment debiased column

{table(["Language", "Row", "E", "V", "S", "D=sign(c)", "OCR+ML E"], f1_rows)}

**Answer:** the stored assets cannot determine whether the strongest
screenshot-only system changes after adding D. The currently measurable
effective winners remain RapidOCR+SVR 52.97% (Java), RapidOCR+GB 59.53%
(Python), and RapidOCR+Linear 60.73% (CUDA). A valid D rerun would require
54,000 calls and must preserve the two-call cost, logit-access/open-model
requirement, and supervised-versus-zero-shot asymmetry.

## F4 both-valid floor

`Original vs E4-A` is a cross-environment comparison. `E4-A vs E4-B` isolates
the GPU change inside the repeat runtime.

{table(["Model", "Lang", "Orig/E4A mismatches", "Cross-env floor", "E4A/E4B mismatches", "GPU-only floor", "Rendering flip range"], f4_rows)}

There is no defensible same-environment grid floor in the stored runs.
Nevertheless, strict-valid decisions are highly stable across the runtime
change: four cells have 0 disagreements, Qwen-Python has 1/393 (0.25%), and
InternVL-Python has 2/429 (0.47%). E4-A and E4-B have zero disagreements in all
six both-valid comparisons, so the physical GPU change adds no observed
effect. This does not rescue the original “same environment” label.

## F3 nondeterminism cause

{table(["Model", "Lang", "Changed calls", "Rate", "|margin| changed", "|margin| same", "|c| changed", "|b| changed", "logit diff med.", "logit diff max", "tie/boundary calls"], f3_rows)}

Changed calls are concentrated at absolute verdict margins of 0.125, versus
0.875--1.000 for unchanged calls. Qwen's changed-call maximum logit difference
is at most 1.000 across cells; InternVL's is 3.125--3.250. The immediate cause
of the earlier contradiction is confirmed as an invalid environment label:
the grid repeat changed Python/runtime and workload, while the exact battery
repeat retained Python 3.13.11. The precise lower-level kernel/library cause
remains unconfirmed; continuous batching is excluded because this is a
single-call direct-Transformers runner. A true original-environment repeat
would require 12,000 new calls; a controlled 3.10-vs-3.13 attribution would
require 24,000 total. Neither was executed.

## F2 Py/CUDA proxy-label agreement ceiling

{table(["Lang", "Difficulty", "Estimator", "Agreement", "95% cluster CI", "Pairs", "Draw-tie probability"], f2_rows)}

## F2 Java within-benchmark ceiling

{table(["Benchmark", "Difficulty", "Estimator", "Agreement", "95% cluster CI", "Used/requested", "Unavailable"], f2j_rows)}

Java cross-dataset pairs: **1,996, undefined**. They join distinct benchmark
panels and scales, so no same-rater or same-panel ceiling was interpolated.

## Table 1 reference check

This table shows the highest battery valid accuracy in each available stratum,
not “human accuracy.”

{table(["Lang", "Benchmark", "Difficulty", "Best model valid", "Panel proxy ceiling", "Ratio"], ref_rows)}

All easy-stratum model values are below the panel proxy-agreement reference.
Where a model approaches or exceeds a finite Monte Carlo reference in another
stratum, it should be described as alignment with a fixed mean-score proxy,
not as superhuman performance.

## Statistical definitions

- `k=1`: exact agreement probability after independently drawing one released
  rating for each snippet; rating ties receive 0.5 credit.
- `k=panel`: 10,000 bootstrap panel means per snippet at that snippet's
  original panel size; ties receive 0.5 credit.
- CI: 10,000 two-endpoint snippet-cluster bootstrap replicates.
- F1 ties (`c=0`) and boundaries (`|c|=|b|`) are unavailable because logits
  were not stored.
- F3/F4 tie and boundary counts are retained separately in the CSV outputs.

## Manuscript actions

1. Table 5 must show D as `not measured`, not borrow D from another run.
2. Replace “same-environment grid noise floor” with “cross-environment
   reproducibility check”; retain the zero GPU-only additional effect.
3. Reconcile Section 6.1's 0.6%--7.1% scope against the complete 66-cell
   reaggregation (0.0%--8.31%) before publication.
4. Call F2 a “proxy-label agreement ceiling,” never “human accuracy,” and
   state that Java cross-dataset is undefined.
"""
    (OUT / "FINAL_F1_F4_REPORT.md").write_text(report, encoding="utf-8")


def build_inventory() -> None:
    inventory_dir = OUT / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    excluded = {
        inventory_dir / "SHA256_INVENTORY.csv",
        inventory_dir / "SHA256_INVENTORY.md",
    }
    rows = []
    for path in sorted(
        p
        for p in OUT.rglob("*")
        if p.is_file()
        and p not in excluded
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    ):
        relative = path.relative_to(OUT)
        section = next((part for part in relative.parts if part in {"F1", "F2", "F3", "F4"}), "package")
        rows.append(
            {
                "run_id": f"{section}_stored_asset_reaggregation_20260731",
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(inventory_dir / "SHA256_INVENTORY.csv", index=False)
    markdown = "# SHA-256 inventory\n\n" + table(
        ["Run ID", "Path", "Bytes", "SHA-256"],
        frame[["run_id", "path", "bytes", "sha256"]].values.tolist(),
    ) + "\n"
    (inventory_dir / "SHA256_INVENTORY.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    build_report()
    build_inventory()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final completeness gate, report builder, and SHA-256 inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
PKG = ROOT / "results/fse2027_review_defense_e1_e8_20260730"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_csv(relative: str, expected_rows: int | None = None) -> pd.DataFrame:
    path = PKG / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(f"{relative}: {len(frame)} rows != {expected_rows}")
    return frame


def main() -> int:
    e1 = require_csv("analysis/E1/E1_debiased_vs_baseline.csv", 15)
    e2 = require_csv("analysis/E2/E2_crossrun_agreement.csv", 6)
    e3 = require_csv("analysis/E3/E3_corrections.csv", 8)
    e4a = require_csv("analysis/E4/E4A_repeat_same_env.csv", 6)
    e4b = require_csv("analysis/E4/E4B_repeat_changed_env.csv", 6)
    e4c = require_csv("analysis/E4/E4C_battery_repeat.csv", 15)
    e5a = require_csv("analysis/E5/E5A_RQ3_TEXT_IMAGE_OVERALL.csv", 2)
    e5b = require_csv("analysis/E5/E5B_INTERNVL_PACKAGING_CELL.csv", 2)
    e6 = require_csv("analysis/E6/E6_deployment_full_metrics.csv", 15)
    e7 = require_csv("analysis/E7/E7_closed_model_pilot.csv", 17)
    e8_contrasts = require_csv("analysis/E8/E8_contrast_inventory.csv", 96)
    e8_robust = require_csv("analysis/E8/E8_label_robustness.csv")

    if not e4c.reproducibility_threshold_pass.isin([True, False]).all():
        raise RuntimeError("E4-C threshold decision missing")
    if (
        e5a.n.sum() != 3600
        or set(e5a.completeness_status) != {"pass", "fail"}
        or e5b.n.iloc[1] != 3000
    ):
        raise RuntimeError("E5 analysis-unit completeness failure")
    if e8_contrasts.holm_p.isna().any():
        raise RuntimeError("E8 adjusted p-values missing")

    conclusion = json.loads((PKG / "analysis/E1/E1_CONCLUSION.json").read_text())
    e3_java = e3[e3["item"] == "E3a Java OCR cost"].iloc[0]
    e3_battery = e3[e3["item"] == "E3b five-model battery recovery range"].iloc[0]
    e6_lines = (PKG / "analysis/E6/E6_CONCLUSION.md").read_text()
    e8_decomp = require_csv("analysis/E8/E8_rendering_decomposition.csv")

    report = f"""# FSE 2027 Review-defense analyses E1-E8

## Scope and integrity

All analyses use immutable stored judgments or isolated new run IDs. GA0 passed
for 234,000 existing grid/battery calls before analysis: pair IDs, AB/BA
allocation, verdict/logit argmax, missingness, duplication, and finite-logit
checks all passed. Existing and repeated runs were never merged. Language-level
results are primary; pooled values are supplementary. Debiased ties (`c=0`) are
incorrect in main estimates, excluded in sensitivity A, and worth 0.5 in
sensitivity B. Strict-validity boundaries (`|c|=|b|`) are counted separately.

## E1: fair VLM-baseline comparison

Conclusion **({conclusion['conclusion']})**: {conclusion['rule']}. Qwen/Python
is the only point-estimate exception, at 66.77% debiased accuracy versus 62.73%
RF and 64.60% language-best Voting. Its two-way snippet-cluster confidence
intervals include zero for both differences. The manuscript must weaken “no VLM
exceeds a baseline in any language”; it may state that no excess is
distinguishable under cluster uncertainty.

## E2: run separation

The battery, clean-default, grid, RQ3, and deployment numbers are separate
runs. Battery and clean-default contain the same 3,000 Java pair IDs. Their
large InternVL swap difference (35.93% versus 54.80%) is therefore not sampling:
the selected-snippet/invalid state differs on
{100*e2[(e2.model.str.contains('InternVL')) & (e2.left_run_id == 'battery') & (e2.right_run_id == 'clean_default')].iloc[0].state_disagreement_rate:.2f}%
of pairs. Run labels must appear in every affected table and figure.

## E3: numerical corrections

- Java OCR loss: {float(e3_java.verified_value):.2f} pp, not 10.5 pp.
- Five-model debiasing recovery: {e3_battery.verified_value}, not 11.7-25.6 pp.
- 108,000 decomposition observations = 72,000 grid + 36,000 perturbation.
  There are 2,946 ties and 8,168 boundaries; 99,832 is the non-boundary count.
- OR 0.55/0.90 models strict-swap invalidity, not correct verdict. On a +0.1 z
  scale the ORs are 0.942 and 0.989.

## E4: reproducibility

Same-environment pair-state disagreement spans
{100*e4a.pair_state_disagreement.min():.2f}-
{100*e4a.pair_state_disagreement.max():.2f}% across six grid cells. Changing
only the physical GPU spans {100*e4b.pair_state_disagreement.min():.2f}-
{100*e4b.pair_state_disagreement.max():.2f}%. The full battery has
{int(e4c.reproducibility_threshold_pass.sum())}/{len(e4c)} cells at or above
the preregistered 99.5% pair-state agreement threshold. Per the fixed rule,
original Table 1/2 values are retained and repeats are reported separately.

## E5: missing cells

RQ3 text+image attempted the full {int(e5a.n.sum())} contrasts and
{2*int(e5a.n.sum())} calls. Qwen passed completeness; InternVL retry 2 retained
one `FINAL_VERD` call without A/B and therefore remains missing rather than
being imputed. InternVL combined-labeled text+image separately completed on
{int(e5b.n.iloc[1])} Java pairs with zero parse failures. The documented
adapter changes parsing only for malformed but unambiguous verdict prefixes.

## E6: deployment

{e6_lines}

## E7: closed-model pilots

The package exposes all existing 300-pair Java closed-model generations and
separates standard 24-token runs from GPT-5.4 low/high 4,096-token runs. These
remain protocol checks, not controlled scale comparisons. The GPT-5.4
high/4,096 cross-modality result reproduces swap error 11.33% to 29.00%,
exact McNemar p=1.18e-7.

## E8: restructuring inputs

All 66 rendering and 30 perturbation contrasts retain effect, cluster CI,
exact McNemar p, Holm p, and direction. Coverage/consistency is the larger
effective-accuracy component in
{int((e8_decomp.dominant_component == 'valid_coverage/consistency').sum())}/
{len(e8_decomp)} cells. On easy-only and Java-within-dataset slices, no VLM
exceeds the best ML baseline. The appearance-only RQ3 contrast remains fragile:
Qwen beautiful-trash versus ugly-trash has 25/300 valid pairs and 8.33%
effective target preference.

## Required manuscript edits

1. Weaken the universal E1 baseline claim and insert the language-specific
   debiased comparison with cluster intervals.
2. Add `battery`, `clean_default`, `grid`, `rq3`, and `deploy` labels; never
   compare their cells as controlled treatments.
3. Correct the four E3 statements exactly as listed above.
4. Add E4 noise-floor and battery reproducibility results without replacing
   original Table 1/2 values.
5. Fill E5 cells and document the output-format adapter and old failures.
6. Replace Table 5 with the denominator-complete E6 table.
7. Add the E7 numeric pilot table with its protocol-check caption.
8. Put the multiplicity family table in Section 3.7 and use the four-row RQ3
   compact view.

## Reproduction

Scripts are under `code/`; immutable results are under `analysis/`; isolated new
runs are under `runs/`; logs and environment snapshots are under `logs/` and
`env/`. `SHA256SUMS` covers every regular package artifact except itself.
"""
    (PKG / "REVIEW_DEFENSE_REPORT.md").write_text(report, encoding="utf-8")

    required_drafts = [
        "analysis/E1/E1_MANUSCRIPT_DRAFT.md",
        "analysis/E2/E2_DISCREPANCY_REPORT.md",
        "analysis/E3/E3_REPORT.md",
        "analysis/E4/E4_REPORT.md",
        "analysis/E4/E4_ENV_DIFF.md",
        "analysis/E5/E5_REPORT.md",
        "analysis/E6/E6_TABLE5_DRAFT.md",
        "analysis/E7/E7_TABLE_DRAFT.md",
        "analysis/E8/E8_REPORT.md",
    ]
    missing = [item for item in required_drafts if not (PKG / item).is_file()]
    if missing:
        raise FileNotFoundError(f"missing report drafts: {missing}")

    manifest = {
        "status": "complete",
        "artifact_count_excluding_sha_file": 0,
        "e1_conclusion": conclusion["conclusion"],
        "e4c_threshold_pass_cells": int(e4c.reproducibility_threshold_pass.sum()),
        "e4c_total_cells": len(e4c),
        "e5a_pairs_attempted": int(e5a.n.sum()),
        "e5a_reportable_cells": int(e5a.reportable.sum()),
        "e5a_failed_cells": int((~e5a.reportable).sum()),
        "e5b_pairs": int(e5b.n.iloc[1]),
    }
    manifest_path = PKG / "FINAL_COMPLETENESS.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    manifest["artifact_count_excluding_sha_file"] = sum(
        path.is_file() and path.name != "SHA256SUMS" for path in PKG.rglob("*")
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    records = []
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            records.append((sha256(path), str(path.relative_to(PKG))))
    (PKG / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for digest, name in records),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

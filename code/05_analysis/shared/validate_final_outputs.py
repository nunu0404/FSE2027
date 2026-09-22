#!/usr/bin/env python3
"""Final machine-readable integrity gate for the grounded experiment package."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors = []
    ab_integrity = json.loads((ROOT / "analysis/ab/INTEGRITY.json").read_text())
    if ab_integrity.get("raw_calls") != 216000 or ab_integrity.get("paired_rows") != 108000:
        errors.append("A/B allocation")
    pair = pd.read_csv(ROOT / "analysis/ab/A_B_PAIR_LEVEL.csv")
    grid = pair[(pair.experiment == "grid") & pair.condition.eq("monokai_dark__fs20__wrap80__lnon")]
    perturb = pair[(pair.experiment == "perturbation") & pair.condition.eq("baseline")]
    merged = grid.merge(perturb, on=["model", "language", "pair_id"], suffixes=("_g", "_b"))
    baseline_exact = len(merged) == 6000 and (
        (merged.margin_ab_g == merged.margin_ab_b) & (merged.margin_ba_g == merged.margin_ba_b)
        & (merged.strict_valid_g == merged.strict_valid_b)
        & (merged.strict_correct_g == merged.strict_correct_b)
    ).all()
    if not baseline_exact:
        errors.append("A/B baseline identity")
    dimensions = {
        "a_grid_rows": len(pd.read_csv(ROOT / "analysis/ab/A_GRID_RESULTS.csv")),
        "b_perturbation_rows": len(pd.read_csv(ROOT / "analysis/ab/B_PERTURBATION_RESULTS.csv")),
        "rq3_image_language_contrast_rows": len(pd.read_csv(
            ROOT / "rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv")),
        "rq3_qwen_text_pair_rows": len(pd.read_csv(
            ROOT / "rq3/analysis/qwen_text_plus_image/RQ3_QWEN_TEXT_PLUS_IMAGE_PAIR_LEVEL.csv")),
        "embedding_stability_rows": len(pd.read_csv(ROOT / "embedding/analysis/d_embedding_stability.csv")),
    }
    expected = {"a_grid_rows": 72, "b_perturbation_rows": 36,
                "rq3_image_language_contrast_rows": 36, "rq3_qwen_text_pair_rows": 1800,
                "embedding_stability_rows": 15}
    if dimensions != expected:
        errors.append("summary dimensions")
    spearman_complete = all(pd.read_csv(ROOT / f"analysis/ab/{name}").spearman_valid_only.notna().all()
                            for name in ("A_GRID_RESULTS.csv", "B_PERTURBATION_RESULTS.csv"))
    if not spearman_complete:
        errors.append("Spearman completeness")
    embedding = []
    for path in sorted((ROOT / "embedding/raw").glob("*__vision__full_20260722.manifest.json")):
        manifest = json.loads(path.read_text())
        embedding.append({"path": str(path.relative_to(ROOT)), "status": manifest.get("status"),
                          "completed_images": manifest.get("completed_images"),
                          "array_shapes": manifest.get("array_shapes")})
    if len(embedding) != 2 or any(x["status"] != "complete" or x["completed_images"] != 9384 for x in embedding):
        errors.append("embedding manifests")
    report = ROOT / "FINAL_GROUNDED_3LANG_REPORT.md"
    if not report.is_file() or report.stat().st_size < 10000:
        errors.append("final report")
    result = {"gate": "FINAL", "gate_pass": not errors, "errors": errors,
              "ab_raw_calls": ab_integrity.get("raw_calls"), "ab_pair_rows": len(pair),
              "baseline_rows": len(merged), "baseline_exact": bool(baseline_exact),
              "spearman_complete": bool(spearman_complete), "dimensions": dimensions,
              "embedding": embedding, "report_bytes": report.stat().st_size,
              "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
    out = ROOT / "analysis/FINAL_VALIDATION.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

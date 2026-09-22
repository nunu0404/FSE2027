#!/usr/bin/env python3
"""Reaggregate F1, F4, and F3 from immutable stored runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ANON/experiment_root")
OLD = ROOT / "results/fse2027_review_defense_e1_e8_20260730"
OUT = ROOT / "results/fse2027_followup_f1_f4_20260731"
GROUND = ROOT / "results/grounded_protocol_3lang_20260721"
BASELINE = "monokai_dark__fs20__wrap80__lnon"
MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B",
}


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_states(calls: pd.DataFrame) -> pd.DataFrame:
    frame = calls.copy()
    frame["selected"] = np.where(
        frame.parsed_choice.eq("A"),
        frame.snippet_first,
        np.where(frame.parsed_choice.eq("B"), frame.snippet_second, None),
    )
    keys = ["model", "language", "pair_id", "snippet_i", "snippet_j"]
    pivot = frame.pivot(index=keys, columns="order", values=["selected", "margin"]).reset_index()
    pivot.columns = [
        "_".join(part for part in col if part) if isinstance(col, tuple) else col
        for col in pivot.columns
    ]
    pivot["valid"] = (
        pivot.selected_AB.notna()
        & pivot.selected_BA.notna()
        & pivot.selected_AB.eq(pivot.selected_BA)
    )
    pivot["decision"] = pivot.selected_AB.where(pivot.valid)
    pivot["c"] = (pivot.margin_AB - pivot.margin_BA) / 2.0
    pivot["b"] = (pivot.margin_AB + pivot.margin_BA) / 2.0
    pivot["tie_c"] = pivot.c.eq(0)
    pivot["boundary"] = pivot.c.abs().eq(pivot.b.abs()) & ~pivot.tie_c
    return pivot


def original_grid(model: str) -> pd.DataFrame:
    safe = model.replace("/", "__")
    path = (
        GROUND
        / "inference/grid/raw"
        / f"{safe}__image_only__promptB__seed42__full_20260721.jsonl"
    )
    return read_jsonl(path)


def repeat_grid(model: str, changed: bool = False) -> pd.DataFrame:
    safe = model.replace("/", "__")
    root = "repeat_changed_env" if changed else "repeat_same_env"
    tag = "e4b_gpu1_20260730" if changed else "e4a_same_env_20260730"
    path = (
        OLD
        / "runs"
        / root
        / "inference/grid/raw"
        / f"{safe}__image_only__promptB__seed42__{tag}.jsonl"
    )
    return read_jsonl(path)


def manifest_path(model: str, repeat: str | None = None) -> Path:
    safe = model.replace("/", "__")
    if repeat is None:
        return (
            GROUND
            / "inference/grid/raw"
            / f"{safe}__image_only__promptB__seed42__full_20260721.manifest.json"
        )
    root = "repeat_same_env" if repeat == "A" else "repeat_changed_env"
    tag = "e4a_same_env_20260730" if repeat == "A" else "e4b_gpu1_20260730"
    return (
        OLD
        / "runs"
        / root
        / "inference/grid/raw"
        / f"{safe}__image_only__promptB__seed42__{tag}.manifest.json"
    )


def analyze_f1() -> None:
    out = OUT / "analysis/F1"
    out.mkdir(parents=True, exist_ok=True)
    e6_path = OLD / "analysis/E6/E6_deployment_full_metrics.csv"
    e6 = pd.read_csv(e6_path)
    neural = e6[
        e6.system.isin(
            [
                "Direct Qwen image-only",
                "Source-text Qwen2.5-Coder",
                "Best OCR-text LLM (RapidOCR)",
            ]
        )
    ].copy()
    source_paths = {
        "java": {
            "Direct Qwen image-only": ROOT
            / "results/screenshot_only_ocr_clean_20260707/clean_direct_vlm_pair_predictions.csv",
            "Source-text Qwen2.5-Coder": ROOT
            / "results/screenshot_only_ocr_clean_20260707/clean_ocr_text_llm_pair_predictions.csv",
            "Best OCR-text LLM (RapidOCR)": ROOT
            / "results/screenshot_only_ocr_clean_20260707/clean_ocr_text_llm_pair_predictions.csv",
        },
        "python": {
            "Direct Qwen image-only": ROOT
            / "results/python_cuda_vlm_main_20260715/outputs/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
            "Source-text Qwen2.5-Coder": ROOT
            / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/source_text_llm.jsonl",
            "Best OCR-text LLM (RapidOCR)": ROOT
            / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/rapidocr_text_llm.jsonl",
        },
        "cuda": {
            "Direct Qwen image-only": ROOT
            / "results/python_cuda_vlm_main_20260715/outputs/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
            "Source-text Qwen2.5-Coder": ROOT
            / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/source_text_llm.jsonl",
            "Best OCR-text LLM (RapidOCR)": ROOT
            / "results/python_cuda_missing_experiments_20260716/ocr_text_llm_raw/rapidocr_text_llm.jsonl",
        },
    }
    ocr = e6[e6.system.eq("RapidOCR + best ML")].set_index("language")
    rows = []
    for row in neural.itertuples(index=False):
        path = source_paths[row.language][row.system]
        if not path.exists() and "rapidocr_text_llm" in path.name:
            alternatives = sorted(path.parent.glob("*rapid*jsonl"))
            if alternatives:
                path = alternatives[0]
        columns: list[str] = []
        if path.suffix == ".csv":
            columns = list(pd.read_csv(path, nrows=1).columns)
        elif path.exists():
            with path.open(encoding="utf-8") as handle:
                first = json.loads(handle.readline())
            columns = list(first)
        has_logits = {"logit_A", "logit_B"}.issubset(columns)
        best = ocr.loc[row.language]
        rows.append(
            {
                "run_id": row.run_id,
                "language": row.language,
                "row": row.system,
                "model": row.model,
                "E": row.effective_accuracy,
                "V": row.valid_accuracy,
                "S": row.strict_swap_error,
                "D_main": np.nan,
                "D_excl": np.nan,
                "D_half": np.nan,
                "ties": np.nan,
                "boundaries": np.nan,
                "diff_vs_ocrml": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "mcnemar_p": np.nan,
                "holm_p": np.nan,
                "n": row.n_total_pairs,
                "ocrml_system": best.system,
                "ocrml_E": best.effective_accuracy,
                "measurement_status": "available" if has_logits else "unavailable_no_verdict_logits",
                "source_asset": str(path.relative_to(ROOT)),
                "source_asset_sha256": sha256(path) if path.exists() else "",
                "stored_columns_include_logits": has_logits,
                "comparison_status": (
                    "not_computable_without_new_inference"
                    if not has_logits
                    else "computable"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "F1_deploy_debiased.csv", index=False)

    table = result[
        ["language", "row", "E", "V", "S", "D_main", "ocrml_E", "measurement_status"]
    ].copy()
    table.to_csv(out / "Table5_final_with_D.csv", index=False)
    draft = """# F1 result and manuscript draft

## Audit conclusion

The stored deployment runs contain generated AB/BA verdicts but not the
verdict-token logits required to compute c=(m_AB-m_BA)/2. Consequently, D is
not measured for Direct Qwen image-only, source-text Qwen2.5-Coder, or
RapidOCR-text Qwen2.5-Coder in any language. Runs with logits elsewhere in the
repository use different rendering or execution protocols and cannot be
merged with the deployment run.

## Required answer

Whether adding the debiased column changes the strongest screenshot-only
system is **not identifiable from the stored deployment assets**. The existing
effective-accuracy result remains: RapidOCR+ML is 52.97% on Java, 59.53% on
Python, and 60.73% on CUDA. It must not be compared against a D value imported
from another run.

## English draft

The deployment runs preserved both order-specific verdicts but did not retain
the verdict-token logits required for logit-level order debiasing. We therefore
report the debiased column as not measured, rather than importing estimates
from a different rendering or execution run. Under the available end-to-end
effective metric, RapidOCR plus the best supervised regressor remains the
strongest screenshot-only pipeline in each language. Establishing whether a
two-call, logit-accessible zero-shot VLM changes that conclusion requires a
separately preregistered deployment rerun; it cannot be inferred from these
stored outputs.

Such a rerun would require 54,000 calls (three neural rows x three languages x
3,000 pairs x two orders), preserve the two-call cost and open-model/logit
access caveats, and retain the supervised OCR+ML versus zero-shot VLM
asymmetry.
"""
    (out / "F1_REPORT.md").write_text(draft, encoding="utf-8")


def rendering_flip_detail(calls: pd.DataFrame) -> pd.DataFrame:
    baseline = pair_states(calls[calls.condition.eq(BASELINE)])
    keys = ["model", "language", "pair_id"]
    rows = []
    for condition in sorted(set(calls.condition) - {BASELINE}):
        current = pair_states(calls[calls.condition.eq(condition)])
        joined = baseline.merge(current, on=keys, suffixes=("_baseline", "_condition"))
        for (model, language), group in joined.groupby(["model", "language"]):
            mask = group.valid_baseline & group.valid_condition
            mismatch = (
                ~group.loc[mask, "decision_baseline"].eq(group.loc[mask, "decision_condition"])
            )
            rows.append(
                {
                    "model": model,
                    "language": language,
                    "condition": condition,
                    "n_denominator": int(mask.sum()),
                    "n_mismatch": int(mismatch.sum()),
                    "flip_bothvalid": float(mismatch.mean()) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def repeat_floor(original: pd.DataFrame, repeat: pd.DataFrame) -> pd.DataFrame:
    left = pair_states(original)
    right = pair_states(repeat)
    keys = ["model", "language", "pair_id"]
    joined = left.merge(right, on=keys, suffixes=("_original", "_repeat"))
    rows = []
    for (model, language), group in joined.groupby(["model", "language"]):
        mask = group.valid_original & group.valid_repeat
        mismatch = ~group.loc[mask, "decision_original"].eq(group.loc[mask, "decision_repeat"])
        rows.append(
            {
                "model": model,
                "language": language,
                "n_denominator": int(mask.sum()),
                "n_mismatch": int(mismatch.sum()),
                "floor": float(mismatch.mean()) if mask.any() else np.nan,
                "ties_original": int(group.tie_c_original.sum()),
                "ties_repeat": int(group.tie_c_repeat.sum()),
                "boundaries_original": int(group.boundary_original.sum()),
                "boundaries_repeat": int(group.boundary_repeat.sum()),
            }
        )
    return pd.DataFrame(rows)


def analyze_f4() -> None:
    out = OUT / "analysis/F4"
    out.mkdir(parents=True, exist_ok=True)
    flip_frames = []
    floor_a = []
    floor_b = []
    a_vs_b = []
    for model in MODELS.values():
        original_all = original_grid(model)
        flip_frames.append(rendering_flip_detail(original_all))
        original = original_all[original_all.condition.eq(BASELINE)].copy()
        repeat_a = repeat_grid(model, changed=False)
        repeat_b = repeat_grid(model, changed=True)
        floor_a.append(repeat_floor(original, repeat_a))
        floor_b.append(repeat_floor(original, repeat_b))
        a_vs_b.append(repeat_floor(repeat_a, repeat_b))
    flips = pd.concat(flip_frames, ignore_index=True)
    flips.to_csv(out / "F4_rendering_bothvalid_flips.csv", index=False)
    a = pd.concat(floor_a, ignore_index=True).rename(
        columns={
            "n_denominator": "n_denominator_e4a",
            "n_mismatch": "n_mismatch_e4a",
            "floor": "floor_e4a_cross_environment",
            "ties_original": "ties_original_e4a_comparison",
            "ties_repeat": "ties_repeat_e4a",
            "boundaries_original": "boundaries_original_e4a_comparison",
            "boundaries_repeat": "boundaries_repeat_e4a",
        }
    )
    b = pd.concat(floor_b, ignore_index=True).rename(
        columns={
            "n_denominator": "n_denominator_e4b",
            "n_mismatch": "n_mismatch_e4b",
            "floor": "floor_changed_env",
            "ties_original": "ties_original_e4b_comparison",
            "ties_repeat": "ties_repeat_e4b",
            "boundaries_original": "boundaries_original_e4b_comparison",
            "boundaries_repeat": "boundaries_repeat_e4b",
        }
    )
    ab = pd.concat(a_vs_b, ignore_index=True).rename(
        columns={
            "n_denominator": "n_denominator_e4a_vs_e4b",
            "n_mismatch": "n_mismatch_e4a_vs_e4b",
            "floor": "floor_e4a_vs_e4b",
            "ties_original": "ties_e4a_in_gpu_comparison",
            "ties_repeat": "ties_e4b_in_gpu_comparison",
            "boundaries_original": "boundaries_e4a_in_gpu_comparison",
            "boundaries_repeat": "boundaries_e4b_in_gpu_comparison",
        }
    )
    summary = (
        flips.groupby(["model", "language"])
        .agg(
            flip_bothvalid_min=("flip_bothvalid", "min"),
            flip_bothvalid_max=("flip_bothvalid", "max"),
            flip_denominator_min=("n_denominator", "min"),
            flip_denominator_max=("n_denominator", "max"),
        )
        .reset_index()
    )
    result = a.merge(b, on=["model", "language"]).merge(ab, on=["model", "language"])
    result = result.merge(summary, on=["model", "language"])
    result.insert(2, "run_id", "grid_original_vs_E4A/E4B_20260730")
    result["floor_same_env"] = np.nan
    result["floor_same_env_status"] = (
        "unavailable_E4A_python_version_differs_from_original"
    )
    result["floor_changed_env_status"] = (
        "original_vs_E4B_combines_runtime_and_GPU_change; E4A_vs_E4B_is_GPU_only"
    )
    result["tie_rule"] = "strict-valid decisions; ties/boundaries remain as stored verdict states"
    result.to_csv(out / "F4_bothvalid_floor.csv", index=False)
    report = f"""# F4 both-valid floor

E4-A cannot supply the requested same-environment floor: the original grid
manifest records Python 3.13.11, whereas E4-A records Python 3.10.12. The
corresponding cross-environment both-valid disagreement ranges from
{100*a.floor_e4a_cross_environment.min():.3f}% to
{100*a.floor_e4a_cross_environment.max():.3f}% across six cells. E4-B produces
the same values relative to the original because E4-A and E4-B are exactly
identical at the both-valid decision level (E4-A versus E4-B floor:
{100*ab.floor_e4a_vs_e4b.min():.3f}% to
{100*ab.floor_e4a_vs_e4b.max():.3f}%).

Across the 11 nonbaseline rendering conditions, the directly recomputed
both-valid flip range is {100*flips.flip_bothvalid.min():.3f}% to
{100*flips.flip_bothvalid.max():.3f}%. The detailed file retains every
condition and denominator.

## English draft

Among pairs that were strict-valid in both rendering conditions, decision
flips ranged from {100*flips.flip_bothvalid.min():.1f}% to
{100*flips.flip_bothvalid.max():.1f}% across model-language-condition cells.
The previously labelled same-environment grid repeat was subsequently found
to use a different Python runtime, so it cannot define a same-environment
noise floor. Its cross-environment both-valid disagreement was at most
{100*a.floor_e4a_cross_environment.max():.1f}%, while changing only the
physical GPU within the repeat environment added no decision disagreement.
We therefore avoid claiming a zero same-environment floor until the baseline
is repeated under the original pinned runtime.
"""
    (out / "F4_REPORT.md").write_text(report, encoding="utf-8")


def qstats(series: pd.Series, prefix: str) -> dict[str, float]:
    valid = series.dropna().astype(float)
    return {
        f"{prefix}_n": int(len(valid)),
        f"{prefix}_median": float(valid.median()) if len(valid) else np.nan,
        f"{prefix}_q25": float(valid.quantile(0.25)) if len(valid) else np.nan,
        f"{prefix}_q75": float(valid.quantile(0.75)) if len(valid) else np.nan,
        f"{prefix}_max": float(valid.max()) if len(valid) else np.nan,
    }


def f3_compare(original: pd.DataFrame, repeat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "language", "pair_id", "order"]
    merged = original.merge(repeat, on=keys, suffixes=("_original", "_repeat"))
    merged["verdict_mismatch"] = ~merged.parsed_choice_original.eq(merged.parsed_choice_repeat)
    merged["abs_margin_original"] = merged.margin_original.abs()
    merged["abs_margin_repeat"] = merged.margin_repeat.abs()
    merged["logit_a_diff"] = (merged.logit_A_original - merged.logit_A_repeat).abs()
    merged["logit_b_diff"] = (merged.logit_B_original - merged.logit_B_repeat).abs()
    merged["max_logit_diff"] = merged[["logit_a_diff", "logit_b_diff"]].max(axis=1)
    op = pair_states(original)[["model", "language", "pair_id", "c", "b", "tie_c", "boundary"]]
    merged = merged.merge(op, on=["model", "language", "pair_id"])
    merged["abs_c"] = merged.c.abs()
    merged["abs_b"] = merged.b.abs()
    rows = []
    for (model, language), group in merged.groupby(["model", "language"]):
        same = group[~group.verdict_mismatch]
        diff = group[group.verdict_mismatch]
        row = {
            "model": model,
            "language": language,
            "n_calls": len(group),
            "mismatch_calls": int(group.verdict_mismatch.sum()),
            "mismatch_rate": float(group.verdict_mismatch.mean()),
            "mismatch_tie_c_calls": int(diff.tie_c.sum()),
            "mismatch_boundary_calls": int(diff.boundary.sum()),
            "all_tie_c_calls": int(group.tie_c.sum()),
            "all_boundary_calls": int(group.boundary.sum()),
        }
        for label, subset in [("mismatch", diff), ("match", same)]:
            row.update(qstats(subset.abs_margin_original, f"{label}_abs_margin"))
            row.update(qstats(subset.abs_c, f"{label}_abs_c"))
            row.update(qstats(subset.abs_b, f"{label}_abs_b"))
        row.update(qstats(diff.max_logit_diff, "mismatch_max_logit_diff"))
        rows.append(row)
    detail_cols = [
        "model",
        "language",
        "pair_id",
        "order",
        "verdict_mismatch",
        "margin_original",
        "margin_repeat",
        "abs_margin_original",
        "logit_A_original",
        "logit_A_repeat",
        "logit_B_original",
        "logit_B_repeat",
        "max_logit_diff",
        "c",
        "b",
        "tie_c",
        "boundary",
    ]
    return pd.DataFrame(rows), merged[detail_cols]


def analyze_f3() -> None:
    out = OUT / "analysis/F3"
    out.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    detail_frames = []
    config_rows = []
    for model in MODELS.values():
        original = original_grid(model)
        original = original[original.condition.eq(BASELINE)].copy()
        repeat = repeat_grid(model, changed=False)
        summary, detail = f3_compare(original, repeat)
        summary_frames.append(summary)
        detail_frames.append(detail)
        manifests = {
            "original_grid": manifest_path(model),
            "E4A": manifest_path(model, "A"),
            "E4B": manifest_path(model, "B"),
        }
        for label, path in manifests.items():
            data = json.loads(path.read_text(encoding="utf-8"))
            config_rows.append(
                {
                    "run": label,
                    "model": model,
                    "manifest": str(path.relative_to(ROOT)),
                    "manifest_sha256": sha256(path),
                    "python": data.get("python"),
                    "cuda_visible_devices": data.get("cuda_visible_devices"),
                    "model_revision": data.get("model_revision"),
                    "prompt_sha256": data.get("prompt_sha256"),
                    "protocol_sha256": data.get("protocol_sha256"),
                    "render_audit_sha256": data.get("render_audit_sha256"),
                    "dtype": data.get("dtype"),
                    "decoding": json.dumps(data.get("decoding"), sort_keys=True),
                    "planned_calls": data.get("planned_calls"),
                    "conditions_n": len(data.get("conditions", [])),
                    "capture_verdict_logits": data.get("capture_verdict_logits"),
                }
            )
    summary = pd.concat(summary_frames, ignore_index=True)
    detail = pd.concat(detail_frames, ignore_index=True)
    configs = pd.DataFrame(config_rows)
    summary.to_csv(out / "F3_mismatch_call_summary.csv", index=False)
    detail.to_csv(out / "F3_mismatch_call_detail.csv", index=False)
    configs.to_csv(out / "F3_manifest_config_diff.csv", index=False)
    min_mismatch_margin = summary.mismatch_abs_margin_median.min()
    max_mismatch_margin = summary.mismatch_abs_margin_median.max()
    min_match_margin = summary.match_abs_margin_median.min()
    max_match_margin = summary.match_abs_margin_median.max()
    report = f"""# F3 grid nondeterminism report

## Setting audit

The original grid and E4-A are not a same-environment repeat. Both model
manifests record Python 3.13.11 for the original grid and Python 3.10.12 for
E4-A. Prompt, model revision, protocol hash, rendering hash, BF16 dtype,
greedy decoding, seed, and GPU index are unchanged. The original run processed
72,000 calls per model over 12 conditions; E4-A processed 6,000 baseline calls
per model. Both runners issue one generation at a time, but the overall request
schedule and runtime environment differ. Package versions, KV-cache policy,
CUDA-graph mode, and deterministic-kernel flags were not fully captured in the
E4-A manifest, so they cannot be asserted equal post hoc.

| Setting | Original grid | E4-A | E4-B | Finding |
|---|---|---|---|---|
| Python | 3.13.11 | 3.10.12 | 3.10.12 | changed before E4-A |
| Calls/model | 72,000 | 6,000 | 6,000 | workload changed |
| Conditions | 12 | baseline only | baseline only | request schedule changed |
| GPU index | 0 | 0 | 1 | E4-B isolates physical GPU within repeat runtime |
| Engine | direct Transformers `generate` | same source runner | same source runner | no serving engine |
| Batching | one generation/call; two model processes concurrent | same | same | no continuous batching |
| TP/PP | none; one GPU/model process | same | same | unchanged |
| Seed | set once/process to 42 | same | same | process-level seed |
| Prompt/model/render | SHA/revision pinned | identical | identical | unchanged |
| KV cache/CUDA graph | not manifest-recorded | not manifest-recorded | not manifest-recorded | unverified |
| Deterministic kernels | no explicit deterministic-algorithm flag in runner | same source | same source | not enforced |
| Processor log | no fast-default warning recorded | Qwen logs a fast-processor default change; InternVL logs slow processor | same repeat stack | library behavior visibly differs |

E4-B uses the same Python 3.10.12 repeat setup as E4-A and changes the physical
GPU. E4-A and E4-B are decision-identical, which excludes the GPU UUID as an
additional observed source under that repeat environment.

## Mismatch characteristics

Across the six cells, verdict-mismatching calls have median original absolute
margin {min_mismatch_margin:.4f}--{max_mismatch_margin:.4f}; matching calls
have median {min_match_margin:.4f}--{max_match_margin:.4f}. Per-cell |c|,
|b|, ties, boundaries, and logit-difference distributions are in
`F3_mismatch_call_summary.csv`; every call is retained in
`F3_mismatch_call_detail.csv`.

## Cause statement

**Confirmed immediate cause of the apparent contradiction:** the experiment
label was wrong; E4-A changed the Python/runtime environment and workload
relative to the original grid, whereas the exact battery repeat retained
Python 3.13.11. The lower-level numerical mechanism is **not confirmed** from
stored metadata. Runtime/library and request-schedule changes can alter BF16
kernel execution near decision boundaries, but attributing the flips
specifically to continuous batching or floating-point accumulation would go
beyond the evidence: this runner is not a continuous-batching server.

## Excluded or unsupported hypotheses

- Physical GPU change: no added effect was observed between E4-A and E4-B.
- Sampling randomness: decoding was greedy with `do_sample=false`.
- Prompt/model/render changes: pinned hashes and model revisions match.
- Continuous batching: not used by the direct Transformers runner.
- Exact package/kernel cause: unresolved because package and kernel flags were
  not recorded in the E4-A manifest.

## Additional inference required, not executed

A true same-environment baseline repeat under the original pinned Python
3.13.11 environment requires 12,000 calls (2 models x 3 languages x 1,000
pairs x 2 orders). A controlled Python 3.10 versus 3.13 attribution would
require another 12,000 calls, for 24,000 total. No such inference was launched.

## English draft for Section 3.6 / Threats

Post hoc manifest auditing showed that the run initially labelled as a
same-environment grid repeat used Python 3.10.12, whereas the original grid
used Python 3.13.11, and also reduced the workload from the 12-condition grid
to the baseline condition. We therefore treat its 3.0--4.4% call disagreement
as cross-environment reproducibility, not a same-environment noise floor.
Changing only the physical GPU within the repeat environment produced no
additional decision disagreement. Because the repeat manifest did not capture
all package and kernel settings, the precise low-level numerical cause remains
unresolved; we do not attribute it specifically to continuous batching.
"""
    (out / "F3_nondeterminism_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    analyze_f1()
    analyze_f4()
    analyze_f3()


if __name__ == "__main__":
    main()

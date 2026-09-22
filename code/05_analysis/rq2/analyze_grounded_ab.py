#!/usr/bin/env python3
"""Aggregate grounded A/B inference and run the preregistered paired analyses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "ab"
BASE_GRID = "monokai_dark__fs20__wrap80__lnon"
BASE_PERT = "baseline"


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_calls() -> pd.DataFrame:
    frames = []
    for experiment in ("grid", "perturbation"):
        for path in sorted((ROOT / "inference" / experiment / "raw").glob("*full*.jsonl")):
            frame = read_jsonl(path)
            frame["raw_file"] = path.name
            frames.append(frame)
    calls = pd.concat(frames, ignore_index=True)
    key = ["experiment", "condition", "model", "language", "pair_id", "order"]
    duplicates = calls.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"duplicate full-run calls: {int(duplicates.sum())}")
    return calls


def pair_calls(calls: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "data" / "pairs_seed42_grounded.csv")
    # Inference pair_id is the protocol identifier (u3_*), not the source dataset pair_id.
    meta = meta.drop(columns=["pair_id"]).rename(columns={"protocol_pair_id": "pair_id"})
    keep = [
        "pair_id", "z_i", "z_j", "human_score_i_z", "human_score_j_z",
        "human_preference", "dataset_name_i",
        "dataset_name_j", "preference_score_basis",
    ]
    rows = []
    group = ["experiment", "condition", "model", "model_revision", "language", "pair_id"]
    for key, part in calls.groupby(group, sort=False, dropna=False):
        by_order = {r.order: r for r in part.itertuples(index=False)}
        if set(by_order) != {"AB", "BA"}:
            raise ValueError(f"missing AB/BA for {key}: {set(by_order)}")
        ab, ba = by_order["AB"], by_order["BA"]

        def chosen_snippet(row):
            if row.parsed_choice == "A":
                return row.snippet_first
            if row.parsed_choice == "B":
                return row.snippet_second
            return None

        selected_ab, selected_ba = chosen_snippet(ab), chosen_snippet(ba)
        parsed = selected_ab is not None and selected_ba is not None
        strict_valid = parsed and selected_ab == selected_ba
        rows.append({
            **dict(zip(group, key)),
            "source_pair_id": ab.source_pair_id,
            "difficulty_rank": ab.difficulty_rank,
            "abs_z_diff": float(ab.abs_z_diff),
            "snippet_i": ab.snippet_i,
            "snippet_j": ab.snippet_j,
            "gold_snippet": ab.snippet_i if ab.gold_side == "first" else ab.snippet_j,
            "selected_ab": selected_ab,
            "selected_ba": selected_ba,
            "parsed_both": parsed,
            "strict_valid": strict_valid,
            "strict_correct": bool(strict_valid and selected_ab == (ab.snippet_i if ab.gold_side == "first" else ab.snippet_j)),
            "margin_ab": float(ab.margin),
            "margin_ba": float(ba.margin),
            "b_position": (float(ab.margin) + float(ba.margin)) / 2,
            "c_content": (float(ab.margin) - float(ba.margin)) / 2,
            "logit_tie_ab": bool(ab.logit_tie),
            "logit_tie_ba": bool(ba.logit_tie),
        })
    paired = pd.DataFrame(rows).merge(meta[keep], on="pair_id", how="left", validate="many_to_one")
    if paired["human_preference"].isna().any():
        raise ValueError("pair metadata join failed")
    # A positive continuous gold difference means snippet_i is preferred.
    paired["gold_diff"] = paired["human_score_i_z"] - paired["human_score_j_z"]
    if paired["gold_diff"].isna().any():
        raise ValueError("canonical human z-score difference is missing")
    paired["model_pair_sign"] = np.where(
        paired.strict_valid,
        np.where(paired.selected_ab == paired.snippet_i, 1.0, -1.0),
        np.nan,
    )
    paired["debiased_selected"] = np.select(
        [paired.c_content > 0, paired.c_content < 0],
        [paired.snippet_i, paired.snippet_j],
        default=None,
    )
    paired["debiased_tie"] = paired.c_content == 0
    paired["debiased_correct"] = (~paired.debiased_tie) & (paired.debiased_selected == paired.human_preference)
    paired["effective_correct"] = paired.strict_correct.astype(int)
    return paired


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def holm_adjust(p_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted < .05, adjusted


def summarize(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in data.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        n, valid = len(part), int(part.strict_valid.sum())
        correct = int(part.strict_correct.sum())
        parse_fail = int((~part.parsed_both).sum())
        eff_lo, eff_hi = wilson(correct, n)
        val_lo, val_hi = wilson(correct, valid)
        rho, rho_p = (math.nan, math.nan)
        v = part.loc[part.strict_valid, ["model_pair_sign", "gold_diff"]].dropna()
        if len(v) > 2 and v.model_pair_sign.nunique() > 1:
            rho, rho_p = spearmanr(v.model_pair_sign, v.gold_diff)
        row = dict(zip(group_cols, key))
        row.update({
            "n_pairs": n,
            "n_valid": valid,
            "n_correct": correct,
            "valid_rate": valid / n,
            "valid_accuracy": correct / valid if valid else math.nan,
            "effective_accuracy": correct / n,
            "strict_swap_error": 1 - valid / n,
            "parse_failure_rate": parse_fail / n,
            "effective_ci_low": eff_lo,
            "effective_ci_high": eff_hi,
            "valid_accuracy_ci_low": val_lo,
            "valid_accuracy_ci_high": val_hi,
            "spearman_valid_only": rho,
            "spearman_p": rho_p,
            "debiased_accuracy": float(part.debiased_correct.mean()),
            "debiased_tie_rate": float(part.debiased_tie.mean()),
            "median_abs_b": float(part.b_position.abs().median()),
            "median_abs_c": float(part.c_content.abs().median()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def exact_mcnemar(data: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for (experiment, model, language), block in data.groupby(["experiment", "model", "language"]):
        wide = block.pivot(index="pair_id", columns="condition", values="effective_correct")
        if baseline not in wide:
            continue
        local = []
        for condition in wide.columns:
            if condition == baseline:
                continue
            both = wide[[baseline, condition]].dropna().astype(int)
            base_only = int(((both[baseline] == 1) & (both[condition] == 0)).sum())
            cond_only = int(((both[baseline] == 0) & (both[condition] == 1)).sum())
            discordant = base_only + cond_only
            p = binomtest(min(base_only, cond_only), discordant, .5).pvalue if discordant else 1.0
            difference = both[condition].to_numpy(dtype=float) - both[baseline].to_numpy(dtype=float)
            pair_meta = block.drop_duplicates("pair_id").set_index("pair_id").loc[both.index]
            residual = difference - difference.mean()
            x = np.ones((len(difference), 1))
            g1 = pair_meta.snippet_i.astype(str).to_numpy()
            g2 = pair_meta.snippet_j.astype(str).to_numpy()
            g12 = np.char.add(np.char.add(g1, "||"), g2)
            meat = (cluster_meat(x, residual, g1) + cluster_meat(x, residual, g2)
                    - cluster_meat(x, residual, g12))
            se = math.sqrt(max(0.0, float(meat[0, 0]))) / len(difference)
            delta = float(difference.mean())
            local.append({
                "experiment": experiment, "model": model, "language": language,
                "baseline": baseline, "condition": condition, "n": len(both),
                "baseline_only_correct": base_only, "condition_only_correct": cond_only,
                "accuracy_delta": delta, "delta_cluster_robust_se": se,
                "delta_ci_low": delta - 1.959963984540054 * se,
                "delta_ci_high": delta + 1.959963984540054 * se,
                "delta_ci_method": "two-way snippet-cluster robust normal interval",
                "mcnemar_exact_p": p,
            })
        if local:
            reject, adjusted = holm_adjust([x["mcnemar_exact_p"] for x in local])
            for item, adj, rej in zip(local, adjusted, reject):
                item["holm_p"] = float(adj)
                item["holm_reject_0_05"] = bool(rej)
            rows.extend(local)
    return pd.DataFrame(rows)


def cochran(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, model, language), block in data.groupby(["experiment", "model", "language"]):
        wide = block.pivot(index="pair_id", columns="condition", values="effective_correct").dropna()
        x = wide.to_numpy(dtype=float)
        k = x.shape[1]
        col = x.sum(axis=0)
        row = x.sum(axis=1)
        numerator = (k - 1) * (k * np.square(col).sum() - col.sum() ** 2)
        denominator = k * col.sum() - np.square(row).sum()
        statistic = numerator / denominator if denominator else math.nan
        rows.append({"experiment": experiment, "model": model, "language": language,
                     "n_pairs": len(wide), "n_conditions": len(wide.columns),
                     "q_statistic": statistic, "df": k - 1,
                     "p": float(chi2.sf(statistic, k - 1)) if np.isfinite(statistic) else math.nan})
    return pd.DataFrame(rows)


def cluster_meat(x: np.ndarray, residual: np.ndarray, groups: np.ndarray) -> np.ndarray:
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in pd.unique(groups):
        idx = groups == group
        score = x[idx].T @ residual[idx]
        meat += np.outer(score, score)
    return meat


def language_interactions(data: pd.DataFrame, dorn_only: bool = True) -> pd.DataFrame:
    """Pairwise baseline contrasts with two-way snippet-cluster robust SEs."""
    rows = []
    perturb = data[data.experiment == "perturbation"]
    if dorn_only:
        perturb = perturb[
            perturb.dataset_name_i.astype(str).str.casefold().eq("dorn")
            & perturb.dataset_name_j.astype(str).str.casefold().eq("dorn")
        ]
    scope = "Dorn-only primary" if dorn_only else "pooled-Java secondary"
    conditions = sorted(set(perturb.condition) - {BASE_PERT})
    for model, model_data in perturb.groupby("model"):
      for endpoint in ("effective_accuracy", "valid_accuracy"):
        endpoint_data = model_data if endpoint == "effective_accuracy" else model_data[model_data.strict_valid]
        for condition in conditions:
            d = endpoint_data[endpoint_data.condition.isin([BASE_PERT, condition])].copy()
            t = (d.condition == condition).astype(float).to_numpy()
            py = (d.language == "python").astype(float).to_numpy()
            cu = (d.language == "cuda").astype(float).to_numpy()
            x = np.column_stack([np.ones(len(d)), py, cu, t, py * t, cu * t])
            y = d.strict_correct.to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            residual = y - x @ beta
            bread = np.linalg.pinv(x.T @ x)
            g1 = d.snippet_i.astype(str).to_numpy()
            g2 = d.snippet_j.astype(str).to_numpy()
            g12 = np.char.add(np.char.add(g1, "||"), g2)
            covariance = bread @ (
                cluster_meat(x, residual, g1) + cluster_meat(x, residual, g2)
                - cluster_meat(x, residual, g12)
            ) @ bread
            se = np.sqrt(np.maximum(np.diag(covariance), 0))
            for idx, language in ((4, "python"), (5, "cuda")):
                z = beta[idx] / se[idx] if se[idx] else math.nan
                rows.append({
                    "analysis_scope": scope, "model": model, "endpoint": endpoint,
                    "condition": condition, "language_vs_java": language,
                    "interaction_effect": beta[idx], "cluster_robust_se": se[idx],
                    "z": z, "p": float(2 * norm.sf(abs(z))) if np.isfinite(z) else math.nan,
                    "n_rows": len(d), "n_pairs": d.pair_id.nunique(),
                    "covariance": "two-way clustered by snippet_i and snippet_j",
                })
    result = pd.DataFrame(rows)
    if len(result):
        result["holm_p_within_model"] = np.nan
        result["holm_reject_0_05"] = False
        for (model, endpoint), idx in result.groupby(["model", "endpoint"]).groups.items():
            reject, adjusted = holm_adjust(result.loc[idx, "p"].tolist())
            result.loc[idx, "holm_p_within_model"] = adjusted
            result.loc[idx, "holm_reject_0_05"] = reject
    return result


def cue_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    render = pd.read_csv(ROOT / "rendered" / "metadata" / "perturbation_render_metadata.csv")
    cues = render.set_index(["rq0_id", "condition"])["cue_present"].to_dict()
    subset = data[(data.experiment == "perturbation") & data.condition.isin(["no_indent", "no_blank_lines"])].copy()
    subset["cue_present_pair"] = [bool(cues.get((i, c), False) or cues.get((j, c), False))
                                  for i, j, c in zip(subset.snippet_i, subset.snippet_j, subset.condition)]
    return summarize(subset, ["model", "language", "condition", "cue_present_pair"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calls = load_calls()
    paired = pair_calls(calls)
    paired.to_csv(OUT / "A_B_PAIR_LEVEL.csv", index=False)
    grid = paired[paired.experiment == "grid"]
    perturb = paired[paired.experiment == "perturbation"]
    summarize(grid, ["model", "language", "condition"]).to_csv(OUT / "A_GRID_RESULTS.csv", index=False)
    summarize(perturb, ["model", "language", "condition"]).to_csv(OUT / "B_PERTURBATION_RESULTS.csv", index=False)
    summarize(paired, ["experiment", "model", "condition"]).to_csv(OUT / "OVERALL_RESULTS.csv", index=False)
    summarize(paired, ["experiment", "model", "language", "difficulty_rank", "condition"]).to_csv(
        OUT / "BY_DIFFICULTY.csv", index=False)
    exact_mcnemar(grid, BASE_GRID).to_csv(OUT / "A_GRID_MCNEMAR.csv", index=False)
    exact_mcnemar(perturb, BASE_PERT).to_csv(OUT / "B_PERTURBATION_MCNEMAR.csv", index=False)
    cochran(paired).to_csv(OUT / "COCHRAN_Q.csv", index=False)
    dorn = paired[
        paired.dataset_name_i.astype(str).str.casefold().eq("dorn")
        & paired.dataset_name_j.astype(str).str.casefold().eq("dorn")
    ]
    summarize(dorn[dorn.experiment.eq("perturbation")], ["model", "language", "condition"]).to_csv(
        OUT / "B_DORN_PRIMARY_RESULTS.csv", index=False)
    primary_interactions = language_interactions(paired, dorn_only=True)
    primary_interactions.to_csv(OUT / "B_LANGUAGE_INTERACTIONS.csv", index=False)
    primary_interactions.to_csv(OUT / "B_LANGUAGE_INTERACTIONS_DORN_PRIMARY.csv", index=False)
    language_interactions(paired, dorn_only=False).to_csv(
        OUT / "B_LANGUAGE_INTERACTIONS_POOLED_SECONDARY.csv", index=False)
    cue_sensitivity(paired).to_csv(OUT / "B_CUE_PRESENT_SENSITIVITY.csv", index=False)

    integrity = {
        "raw_calls": len(calls), "paired_rows": len(paired),
        "expected_raw_calls": 216000, "expected_paired_rows": 108000,
        "all_ab_ba_complete": len(calls) == 216000 and len(paired) == 108000,
        "parse_failures": int((~paired.parsed_both).sum()),
        "models": sorted(paired.model.unique()), "languages": sorted(paired.language.unique()),
        "grid_conditions": sorted(grid.condition.unique()),
        "perturbation_conditions": sorted(perturb.condition.unique()),
    }
    (OUT / "INTEGRITY.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()

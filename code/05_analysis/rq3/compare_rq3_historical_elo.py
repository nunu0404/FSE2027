#!/usr/bin/env python3
"""Compare new direct-pairwise RQ3 ordering with historical Java Swiss/Elo ordering."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PAIR = ROOT / "rq3" / "analysis" / "image_only" / "RQ3_IMAGE_ONLY_PAIR_LEVEL.csv"
OLD = WORKSPACE / "2026-01-23_adversarial_aesthetic_bias/results_adversarial/experiment_run/mm_image/elo_scores.csv"
OUT = ROOT / "rq3" / "analysis" / "historical_alignment"
VARIANTS = ["golden", "ugly_gold", "beautiful_trash", "ugly_trash"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = pd.read_csv(PAIR)
    valid = new[new.strict_valid].copy()
    valid["selected_variant"] = valid.ab_choice
    new["debiased_selected_variant"] = new.preference_target_variant.where(
        new.debiased_target_selected,
        new.variant_j.where(new.variant_i.eq(new.preference_target_variant), new.variant_i),
    )
    rows = []
    for (model, language), data in new.groupby(["model", "language"]):
        vd = valid[(valid.model == model) & (valid.language == language)]
        for variant in VARIANTS:
            strict_games = vd[vd.variant_i.eq(variant) | vd.variant_j.eq(variant)]
            deb_games = data[data.variant_i.eq(variant) | data.variant_j.eq(variant)]
            rows.append({
                "source": "new_direct_pairwise", "model": model, "language": language,
                "variant": variant, "n_strict_valid_games": len(strict_games),
                "strict_valid_win_rate": strict_games.selected_variant.eq(variant).mean(),
                "n_debiased_games": len(deb_games),
                "debiased_win_rate": deb_games.debiased_selected_variant.eq(variant).mean(),
            })
    direct = pd.DataFrame(rows)
    direct["strict_rank"] = direct.groupby(["model", "language"])["strict_valid_win_rate"].rank(
        ascending=False, method="min")
    direct["debiased_rank"] = direct.groupby(["model", "language"])["debiased_win_rate"].rank(
        ascending=False, method="min")
    direct.to_csv(OUT / "RQ3_DIRECT_VARIANT_RANKING.csv", index=False)

    old = pd.read_csv(OLD)
    old["variant"] = old.snippet_id.str.extract(r"_(golden|ugly_gold|beautiful_trash|ugly_trash)$")[0]
    historical = old.groupby("variant", as_index=False).agg(n_items=("snippet_id", "size"), mean_elo=("elo_score", "mean"))
    historical["elo_rank"] = historical.mean_elo.rank(ascending=False, method="min")
    historical.to_csv(OUT / "HISTORICAL_JAVA_IMAGE_ELO.csv", index=False)

    hist_order = ">".join(historical.sort_values(["elo_rank", "variant"]).variant)
    align = []
    for (model, language), d in direct.groupby(["model", "language"]):
        strict_order = ">".join(d.sort_values(["strict_rank", "variant"]).variant)
        deb_order = ">".join(d.sort_values(["debiased_rank", "variant"]).variant)
        align.append({"model": model, "language": language, "historical_order": hist_order,
                      "strict_order": strict_order, "strict_exact_order_match": strict_order == hist_order,
                      "debiased_order": deb_order, "debiased_exact_order_match": deb_order == hist_order})
    pd.DataFrame(align).to_csv(OUT / "RQ3_ORDER_ALIGNMENT.csv", index=False)
    status = pd.DataFrame([
        {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "modality": "image_only", "ga0": "PASS",
         "full_calls": 3600, "status": "complete", "reason": ""},
        {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "modality": "text_plus_image", "ga0": "PASS",
         "full_calls": 3600, "status": "complete", "reason": ""},
        {"model": "OpenGVLab/InternVL3-8B", "modality": "image_only", "ga0": "PASS",
         "full_calls": 3600, "status": "complete", "reason": ""},
        {"model": "OpenGVLab/InternVL3-8B", "modality": "text_plus_image", "ga0": "FAIL twice",
         "full_calls": 0, "status": "blocked_by_gate",
         "reason": "same deterministic malformed verdict; each 100-call pilot had 1 parse failure and 1 argmax mismatch"},
    ])
    status.to_csv(OUT / "RQ3_ARM_STATUS.csv", index=False)
    (OUT / "LIMITATIONS.md").write_text(
        "# Historical alignment limitations\n\n"
        "The historical Swiss/Elo and new direct-pairwise runs differ in sampling, renderer, "
        "variant generator, model provenance, and aggregation. Numeric effect sizes are not pooled. "
        "Only ordinal direction is compared. The historical report states 200 bases, while the "
        "available image-only Elo file contains 160 items per variant; the artifact count is reported.\n",
        encoding="utf-8")
    print(pd.DataFrame(align).to_string(index=False))


if __name__ == "__main__":
    main()

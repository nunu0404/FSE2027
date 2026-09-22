#!/usr/bin/env python3
"""Estimate proxy-label agreement ceilings from public rater matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/fse2027_followup_f1_f4_20260731"
RAW = ROOT / "data/raw_ratings/official_20260713/extracted"
PAIRS = ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
SNIPPETS = ROOT / "experiments/rq0_viability/data/processed/pooled_313_processed.csv"
EXT_SNIPPETS = ROOT / "results/python_cuda_vlm_main_20260715/data/snippets.csv"
BATTERY = ROOT / "results/rq1_model_battery_3lang_20260723"
SEED = 42
PANEL_DRAWS = 10_000
BOOTSTRAP_REPS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_rating_vectors() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    vectors: dict[str, np.ndarray] = {}
    audit = []

    ext = pd.read_csv(EXT_SNIPPETS)
    dorn_root = RAW / "DatasetDorn/dataset/scores"
    for language in ("python", "cuda"):
        path = dorn_root / f"{language}.csv"
        raw = pd.read_csv(path, header=None)
        matrix = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        snippets = ext[ext.language.eq(language)]
        for row in snippets.itertuples(index=False):
            values = matrix.iloc[:, int(row.original_snippet_id)].dropna().to_numpy(float)
            if not np.isclose(values.mean(), row.human_mean_score, atol=1e-10):
                raise RuntimeError(f"{row.rq0_id}: public ratings do not reproduce proxy")
            vectors[str(row.rq0_id)] = values
        audit.append(
            {
                "benchmark": "Dorn",
                "language": language,
                "rating_asset": str(path.relative_to(ROOT)),
                "rating_asset_sha256": sha256(path),
                "mapped_snippets": len(snippets),
                "unmapped_snippets": 0,
                "mapping": "rating column = original_snippet_id; participant-id first column",
            }
        )

    pooled = pd.read_csv(SNIPPETS)
    buse_path = RAW / "DatasetBW/oracle.csv"
    buse = pd.read_csv(buse_path, header=None).iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    subset = pooled[pooled.dataset_name.eq("Buse")]
    for row in subset.itertuples(index=False):
        values = buse.iloc[:, int(row.snippet_id) - 1].dropna().to_numpy(float)
        if not np.isclose(values.mean(), row.human_mean_score, atol=1e-10):
            raise RuntimeError(f"{row.rq0_id}: Buse ratings do not reproduce proxy")
        vectors[str(row.rq0_id)] = values
    audit.append(
        {
            "benchmark": "Buse",
            "language": "java",
            "rating_asset": str(buse_path.relative_to(ROOT)),
            "rating_asset_sha256": sha256(buse_path),
            "mapped_snippets": len(subset),
            "unmapped_snippets": 0,
            "mapping": "100 score columns map to snippet_id 1..100; two metadata columns",
        }
    )

    scal_path = RAW / "Dataset/Dataset/scores.csv"
    scal = pd.read_csv(scal_path).iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    subset = pooled[pooled.dataset_name.eq("Scalabrino")]
    for row in subset.itertuples(index=False):
        values = scal.iloc[:, int(row.snippet_id) - 1].dropna().to_numpy(float)
        if not np.isclose(values.mean(), row.human_mean_score, atol=1e-10):
            raise RuntimeError(f"{row.rq0_id}: Scalabrino ratings do not reproduce proxy")
        vectors[str(row.rq0_id)] = values
    audit.append(
        {
            "benchmark": "Scalabrino",
            "language": "java",
            "rating_asset": str(scal_path.relative_to(ROOT)),
            "rating_asset_sha256": sha256(scal_path),
            "mapped_snippets": len(subset),
            "unmapped_snippets": 0,
            "mapping": "Snippet1..Snippet200 columns map to snippet_id 1..200",
        }
    )

    dorn_path = RAW / "DatasetDorn/dataset/scores/java.csv"
    dorn = pd.read_csv(dorn_path, header=None).iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    subset = pooled[pooled.dataset_name.eq("Dorn")]
    mapped = 0
    missing = []
    # The distributed matrix reproduces the existing Java proxy for IDs
    # 102..221 at zero-based column snippet_id-102. ID 101 has no preceding
    # column and cannot be reconstructed without inventing a mapping.
    for row in subset.itertuples(index=False):
        column = int(row.snippet_id) - 102
        if column < 0 or column >= dorn.shape[1]:
            missing.append(str(row.rq0_id))
            continue
        values = dorn.iloc[:, column].dropna().to_numpy(float)
        if not np.isclose(values.mean(), row.human_mean_score, atol=5e-7):
            raise RuntimeError(
                f"{row.rq0_id}: Dorn Java candidate ratings do not reproduce proxy"
            )
        vectors[str(row.rq0_id)] = values
        mapped += 1
    audit.append(
        {
            "benchmark": "Dorn",
            "language": "java",
            "rating_asset": str(dorn_path.relative_to(ROOT)),
            "rating_asset_sha256": sha256(dorn_path),
            "mapped_snippets": mapped,
            "unmapped_snippets": len(missing),
            "mapping": (
                "for proxy IDs 102..221, zero-based rating column = snippet_id-102; "
                "rq0_0099/snippet 101 has no reproducible public-rating column"
            ),
            "unmapped_ids": ";".join(missing),
        }
    )
    return vectors, pd.DataFrame(audit)


def single_rater_probability(a: np.ndarray, b: np.ndarray, gold_sign: int) -> float:
    differences = a[:, None] - b[None, :]
    credit = np.where(np.sign(differences) == gold_sign, 1.0, np.where(differences == 0, 0.5, 0.0))
    return float(credit.mean())


def precompute_panel_means(
    vectors: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    means = {}
    for snippet, values in vectors.items():
        n = len(values)
        sums = np.zeros(PANEL_DRAWS, dtype=np.float64)
        # Chunk over panel members to avoid allocating draws x panel size.
        for _ in range(n):
            sums += values[rng.integers(0, n, size=PANEL_DRAWS)]
        means[snippet] = sums / n
    return means


def pair_probabilities(
    pairs: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    panel_means: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for row in pairs.itertuples(index=False):
        i, j = str(row.snippet_i), str(row.snippet_j)
        available = i in vectors and j in vectors
        result = {
            "protocol_pair_id": row.protocol_pair_id,
            "language": row.language,
            "benchmark": (
                row.dataset_name_i if row.dataset_name_i == row.dataset_name_j else "cross-dataset"
            ),
            "difficulty": row.difficulty,
            "snippet_i": i,
            "snippet_j": j,
            "available": available,
            "unavailable_reason": "" if available else "one_or_both_rater_vectors_unmapped",
        }
        if available:
            gold_sign = int(np.sign(row.human_score_i_z - row.human_score_j_z))
            if gold_sign == 0:
                raise RuntimeError(f"proxy tie in pair {row.protocol_pair_id}")
            result["k1_agreement"] = single_rater_probability(
                vectors[i], vectors[j], gold_sign
            )
            differences = panel_means[i] - panel_means[j]
            result["panel_agreement"] = float(
                np.where(
                    np.sign(differences) == gold_sign,
                    1.0,
                    np.where(differences == 0, 0.5, 0.0),
                ).mean()
            )
            result["k1_tie_probability"] = float(
                (vectors[i][:, None] == vectors[j][None, :]).mean()
            )
            result["panel_tie_probability"] = float((differences == 0).mean())
            result["panel_size_i"] = len(vectors[i])
            result["panel_size_j"] = len(vectors[j])
        rows.append(result)
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(
    pairs: pd.DataFrame, value_column: str, rng: np.random.Generator
) -> tuple[float, float]:
    snippets = pd.Index(sorted(set(pairs.snippet_i) | set(pairs.snippet_j)))
    index = {snippet: k for k, snippet in enumerate(snippets)}
    i_index = pairs.snippet_i.map(index).to_numpy()
    j_index = pairs.snippet_j.map(index).to_numpy()
    values = pairs[value_column].to_numpy(float)
    estimates = np.empty(BOOTSTRAP_REPS, dtype=float)
    batch = 200
    for start in range(0, BOOTSTRAP_REPS, batch):
        stop = min(start + batch, BOOTSTRAP_REPS)
        counts = rng.multinomial(
            len(snippets),
            np.full(len(snippets), 1.0 / len(snippets)),
            size=stop - start,
        )
        weights = counts[:, i_index] * counts[:, j_index]
        denominator = weights.sum(axis=1)
        estimates[start:stop] = np.divide(
            (weights * values).sum(axis=1),
            denominator,
            out=np.full(stop - start, np.nan),
            where=denominator > 0,
        )
    return tuple(np.nanquantile(estimates, [0.025, 0.975]))


def summarize_group(
    group: pd.DataFrame,
    grouping: dict[str, str],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows = []
    for estimator, column in [
        ("k=1", "k1_agreement"),
        ("k=panel", "panel_agreement"),
    ]:
        lo, hi = cluster_bootstrap_ci(group, column, rng)
        tie_column = "k1_tie_probability" if estimator == "k=1" else "panel_tie_probability"
        rows.append(
            {
                **grouping,
                "estimator": estimator,
                "agreement_probability": float(group[column].mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_pairs": len(group),
                "n_unique_snippets": len(set(group.snippet_i) | set(group.snippet_j)),
                "mean_tie_probability": float(group[tie_column].mean()),
                "tie_credit": 0.5,
                "panel_draws_per_snippet": PANEL_DRAWS if estimator == "k=panel" else 0,
                "cluster_bootstrap_reps": BOOTSTRAP_REPS,
                "cluster_unit": "both snippet endpoints (two-way multiplicity bootstrap)",
            }
        )
    return rows


def load_battery_valid_accuracy(pairs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model_dir in ("qwen", "internvl", "gemma", "ministral", "phi"):
        path = BATTERY / "inference/full" / model_dir / "raw.jsonl"
        data = pd.read_json(path, lines=True)
        if len(data) != 18_000:
            raise RuntimeError(f"incomplete formal battery run: {path} ({len(data)} calls)")
        data["selected"] = np.where(
            data.parsed_choice.eq("A"),
            data.snippet_first,
            np.where(data.parsed_choice.eq("B"), data.snippet_second, None),
        )
        pivot = data.pivot(
            index=["model", "language", "pair_id"],
            columns="order",
            values="selected",
        ).reset_index()
        pivot["valid"] = (
            pivot.AB.notna() & pivot.BA.notna() & pivot.AB.eq(pivot.BA)
        )
        pivot["decision"] = pivot.AB.where(pivot.valid)
        frames.append(pivot)
    states = pd.concat(frames, ignore_index=True)
    pair_meta = pairs[
        [
            "protocol_pair_id",
            "language",
            "difficulty",
            "human_preference",
            "dataset_name_i",
            "dataset_name_j",
        ]
    ].rename(columns={"protocol_pair_id": "pair_id"})
    states = states.merge(pair_meta, on=["pair_id", "language"], validate="many_to_one")
    states["benchmark"] = np.where(
        states.dataset_name_i.eq(states.dataset_name_j),
        states.dataset_name_i,
        "cross-dataset",
    )
    states = states[
        ~states.language.eq("java") | ~states.benchmark.eq("cross-dataset")
    ].copy()
    states["correct"] = states.decision.eq(states.human_preference)
    rows = []
    for (model, language, benchmark, difficulty), group in states.groupby(
        ["model", "language", "benchmark", "difficulty"]
    ):
        valid = group[group.valid]
        rows.append(
            {
                "model": model,
                "language": language,
                "benchmark": benchmark,
                "difficulty": difficulty,
                "n_total": len(group),
                "n_valid": len(valid),
                "valid_accuracy": float(valid.correct.mean()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out = OUT / "analysis/F2"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    vectors, audit = load_rating_vectors()
    audit.to_csv(out / "F2_rating_mapping_audit.csv", index=False)
    pairs = pd.read_csv(PAIRS)
    panel_means = precompute_panel_means(vectors, rng)
    probabilities = pair_probabilities(pairs, vectors, panel_means)
    probabilities.to_csv(out / "F2_pair_probabilities.csv", index=False)

    language_rows = []
    for (language, difficulty), group in probabilities[
        probabilities.language.isin(["python", "cuda"]) & probabilities.available
    ].groupby(["language", "difficulty"]):
        language_rows.extend(
            summarize_group(
                group,
                {"run_id": "rq1_pairs_9000_seed42", "language": language, "difficulty": difficulty},
                rng,
            )
        )
    language_result = pd.DataFrame(language_rows)
    language_result.to_csv(out / "F2_noise_ceiling.csv", index=False)

    java = probabilities[
        probabilities.language.eq("java") & ~probabilities.benchmark.eq("cross-dataset")
    ]
    java_rows = []
    for (benchmark, difficulty), raw_group in java.groupby(["benchmark", "difficulty"]):
        available = raw_group[raw_group.available]
        rows = summarize_group(
            available,
            {
                "run_id": "rq1_java_within_seed42",
                "benchmark": benchmark,
                "difficulty": difficulty,
            },
            rng,
        )
        for row in rows:
            row["n_pairs_requested"] = len(raw_group)
            row["n_pairs_unavailable"] = int((~raw_group.available).sum())
        java_rows.extend(rows)
    for benchmark, raw_group in java.groupby("benchmark"):
        available = raw_group[raw_group.available]
        rows = summarize_group(
            available,
            {
                "run_id": "rq1_java_within_seed42",
                "benchmark": benchmark,
                "difficulty": "all",
            },
            rng,
        )
        for row in rows:
            row["n_pairs_requested"] = len(raw_group)
            row["n_pairs_unavailable"] = int((~raw_group.available).sum())
        java_rows.extend(rows)
    java_result = pd.DataFrame(java_rows)
    java_result.to_csv(out / "F2_java_within.csv", index=False)

    cross = probabilities[
        probabilities.language.eq("java") & probabilities.benchmark.eq("cross-dataset")
    ]
    pd.DataFrame(
        [
            {
                "run_id": "rq1_java_cross_seed42",
                "language": "java",
                "scope": "cross-dataset",
                "n_pairs": len(cross),
                "status": "undefined",
                "reason": (
                    "ratings from distinct benchmark panels/scales cannot define a "
                    "single-rater or same-panel agreement ceiling; no interpolation used"
                ),
            }
        ]
    ).to_csv(out / "F2_java_cross_undefined.csv", index=False)

    valid = load_battery_valid_accuracy(pairs)
    ceilings_language = language_result.pivot_table(
        index=["language", "difficulty"],
        columns="estimator",
        values="agreement_probability",
    ).reset_index().rename(columns={"k=1": "ceiling_k1", "k=panel": "ceiling_panel"})
    ceilings_language["benchmark"] = "Dorn"
    ceilings_java = java_result[java_result.difficulty.ne("all")].pivot_table(
        index=["benchmark", "difficulty"],
        columns="estimator",
        values="agreement_probability",
    ).reset_index().rename(columns={"k=1": "ceiling_k1", "k=panel": "ceiling_panel"})
    ceilings_java["language"] = "java"
    ceilings = pd.concat(
        [
            ceilings_language[
                ["language", "benchmark", "difficulty", "ceiling_k1", "ceiling_panel"]
            ],
            ceilings_java[
                ["language", "benchmark", "difficulty", "ceiling_k1", "ceiling_panel"]
            ],
        ],
        ignore_index=True,
    )
    table1 = valid.merge(
        ceilings, on=["language", "benchmark", "difficulty"], how="left"
    )
    table1["valid_accuracy_over_panel_ceiling"] = (
        table1.valid_accuracy / table1.ceiling_panel
    )
    table1.to_csv(out / "F2_table1_reference.csv", index=False)

    easy_max = (
        table1[table1.difficulty.eq("easy")]
        .groupby("language")
        .valid_accuracy.max()
        .to_dict()
    )
    easy_ceiling = (
        ceilings[
            ceilings.difficulty.eq("easy") & ceilings.language.isin(["python", "cuda"])
        ]
        .set_index("language")
        .ceiling_panel.to_dict()
    )
    report = f"""# F2 proxy-label agreement ceiling

## Definition

This is not human accuracy. It is the probability that a resampled individual
rating (`k=1`) or a bootstrap panel mean at each snippet's original panel size
(`k=panel`) agrees with the direction of the stored mean-score proxy label.
Ratings for the two snippets are sampled independently from their observed
rating distributions because the full pair set is not restricted to common
raters. Draw ties receive 0.5 credit. Confidence intervals use 10,000
two-endpoint snippet-cluster bootstrap replicates.

## Coverage and mapping caveat

Python and CUDA cover all 3,000 pairs per language. Java covers the 1,004
within-dataset pairs benchmark by benchmark; the 1,996 cross-dataset pairs are
undefined and are not interpolated. The public Dorn Java matrix reproduces the
stored proxy means for 89/90 selected snippets using its distributed column
layout. It has no column that reproduces `rq0_0099` (snippet 101), so the nine
within-Dorn pairs containing that snippet are reported unavailable rather than
assigned an invented rating vector.

## Sanity check

For the five-model battery, the highest easy-bin valid accuracy is
{100*easy_max.get('python', float('nan')):.2f}% on Python and
{100*easy_max.get('cuda', float('nan')):.2f}% on CUDA. The corresponding
bootstrap-panel proxy-agreement ceilings are
{100*easy_ceiling.get('python', float('nan')):.2f}% and
{100*easy_ceiling.get('cuda', float('nan')):.2f}%. Any model value above the
estimated ceiling should be interpreted as alignment with this fixed proxy
sample, not as exceeding human performance.

## English Threats draft

We estimated a proxy-label agreement ceiling by resampling the released rater
matrices. For k=1, we computed the probability that one independently sampled
rating per snippet reproduced the direction of the benchmark mean-score
label; for k=panel, we bootstrap-resampled each snippet at its original panel
size and compared the resulting means. Ties received half credit, and 95%
intervals used 10,000 two-endpoint snippet-cluster bootstrap replicates. These
quantities are agreement probabilities with the proxy label, not estimates of
human accuracy. Java cross-benchmark pairs have no coherent shared-panel
ceiling and were left undefined. In addition, one selected Dorn Java snippet
could not be mapped to a released rating column that reproduced its stored
mean, so its nine within-Dorn pairs were explicitly excluded.
"""
    (out / "F2_REPORT.md").write_text(report, encoding="utf-8")

    metadata = {
        "run_id": "F2_stored_rater_reaggregation_20260731",
        "seed": SEED,
        "panel_draws": PANEL_DRAWS,
        "cluster_bootstrap_reps": BOOTSTRAP_REPS,
        "new_model_inference_calls": 0,
        "pair_asset": str(PAIRS.relative_to(ROOT)),
        "pair_asset_sha256": sha256(PAIRS),
    }
    (out / "F2_METHOD.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

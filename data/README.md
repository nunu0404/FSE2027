# Data

## `benchmarks_raw/downloads/`

The published benchmark archives as downloaded, with `SHA256SUMS`.

| Archive | Benchmark | Used by the paper |
| --- | --- | --- |
| `DatasetBW.zip` | Buse and Weimer [7] | yes — 99 Java snippets |
| `DatasetDorn.zip` | Dorn [14] | yes — 90 Java, 119 Python, 120 CUDA |
| `Dataset.zip` | Scalabrino et al. [51] | yes — 124 Java snippets |
| `sergeyuk_readability.zip` | Sergeyuk et al. | **no** — see `EXCLUSIONS.md` |

Parsed and standardized by `code/00_data_preparation/prepare_datasets.py`.

## `snippets/`

| File | Rows | Contents |
| --- | --- | --- |
| `snippets_552_with_human_ratings.csv` | 552 | `rq0_id`, `dataset_name`, `language`, `raw_code`, `human_mean_score`, `human_score_z_within_dataset`, `human_score_z_within_language`, `formatting_code_hash` |
| `rq3_base_snippets_300_seed42.csv` | 300 | The RQ3 base snippets, 100 per language |

Composition: Java 313, Python 119, CUDA 120; Buse 99, Dorn 329, Scalabrino 124.

Ratings are the published means, standardized within each benchmark. The
`human_score_z_within_dataset` column defines the reference direction for a
pair: the higher value is the preferred snippet.

## `pairs/`

| File | Rows | Used by |
| --- | --- | --- |
| `rq1_pairs_9000_seed42.csv` | 9,000 | RQ1, RQ4 |
| `rq2_pairs_3000_seed42.csv` | 3,000 | RQ2 (1,000 per language) |
| `rq1_ga0_gate_pairs_50_seed42.csv` | 50 | Pre-run gate |

`rq1_pairs_9000_seed42.csv` carries both endpoints' snippet ids, benchmark
names, standardized scores, `abs_z_diff`, the rating-derived `human_preference`,
the stratum label, and the two image paths with their SHA-256 digests.

Strata (`difficulty`): `hard` 0.2 ≤ |Δz| < 0.5, `medium` 0.5 ≤ |Δz| < 1.0,
`easy` |Δz| ≥ 1.0. 1,000 pairs per language per stratum. Pairs with |Δz| < 0.2
are excluded, so this is an evaluation distribution rather than a natural one.
Because pairs share snippets, the 9,000 rows are dependent observations; all
intervals in the paper resample snippet clusters, not rows.

## `rq3_variants/`

The four RQ3 variants and the six contrasts between them.

| File | Contents |
| --- | --- |
| `rq3_variant_sources.csv` | Transformed source for every variant |
| `rq3_mutation_log.csv` | Per-mutation record (5–333 per snippet, ≥2 transformation families) |
| `rq3_contrast_manifest.csv` | The six contrasts, with the reference variant for each |
| `rq3_render_metadata.csv` | Per-image render parameters and SHA-256 |
| `rq3_grounded_bases_seed42.csv` | The 300 base snippets |

Variant naming differs between the manuscript and these files:
`Beautiful_gold` = `golden`, `Beautiful_trash` = `beautiful_trash`,
`Ugly_gold` = `ugly_gold`, `Ugly_trash` = `ugly_trash`.

## `features_and_baseline_predictions/`

Source-feature predictors — the external reference that reads the code text.

| Path | Contents |
| --- | --- |
| `ml_predictions_9models_9000.csv` | Out-of-fold pairwise predictions for all nine predictors on all 9,000 pairs: one row per pair × model, with `is_correct`, both predicted scores, and `prediction_tie`. The `source_file` column records which upstream run each row came from. **Stale for three pairs — see the warning below before using it for any reported number** |
| `java/features_313.csv` | 27 source features for the 313 Java snippets |
| `java/{buse,dorn,scalabrino}_processed.csv` | Per-benchmark parsed snippets and ratings |
| `java/split_assignments.csv` | Five-fold multiple-source grouped CV assignments |
| `java/canonical_pair_level.csv`, `classical_*` | Java predictor outputs |
| `java/table1_ml_replication/` | **The value of record for the three per-language best predictors.** `canonical_pairwise_summary.csv` (9 cells with exact numerators), `canonical_pairwise_predictions.csv` (27,000 decisions), `legacy_reported_values_audit.csv` (stale vs corrected), `pair_id_reuse_audit.csv`, `SHA256SUMS`. Duplicated at `results/rq1_viability/analysis_5model_battery/table1_ml_replication/` |
| `python_cuda/source_features_27.csv` | Features for the Dorn Python/CUDA snippets (and, unused here, Sergeyuk rows) |
| `python_cuda/pair_predictions.csv` | Python/CUDA pairwise predictions |
| `python_cuda/oof_predictions.csv` | Out-of-fold snippet scores |

### Which file to use for a reported baseline accuracy

> **Use `java/table1_ml_replication/canonical_pairwise_summary.csv`, not
> `ml_predictions_9models_9000.csv`.**

A clean-pair rebuild reused three `pair_id`s (one CUDA, two Python) after their
endpoint snippets changed. `ml_predictions_9models_9000.csv` inherits those
three stale rows from its upstream source and scores them against the old
endpoints, so it under-counts by up to 2 correct pairs per language.

| Language | Best predictor | Corrected (value of record) | Stale 9-model file |
| --- | --- | --- | --- |
| Java | Multilayer Perceptron | 1895/3000 = **63.17%** | 1895/3000 = 63.17% |
| Python | Voting ensemble (LR+NB+RF) | 1940/3000 = **64.67%** | 1938/3000 = 64.60% |
| CUDA | SVR | 2274/3000 = **75.80%** | 2274/3000 = 75.80% |

Equally weighted reference: **67.88%**, the manuscript's 67.9%.

Best-predictor selection is unchanged in all three languages. `legacy_reported_values_audit.csv`
gives the stale value, the corrected value and the delta for all nine cells;
`pair_id_reuse_audit.csv` lists the three pairs.

**No VLM result is affected.** `data/pairs/rq1_pairs_9000_seed42.csv` carries the
frozen endpoints, so every judge saw the correct images. The nine-predictor file
is kept because it is the only one covering all nine predictors. Full account:
"A trap for replicators" in `../CLAIMS_TO_ARTIFACTS.md`.

These predictions are snippet-level out-of-fold, not fully endpoint-disjoint:
across folds an endpoint can train its partner's predictor. Section 4.2 of the
paper states this limitation, and RQ4 inherits it.

## `render_metadata/`

| File | Contents |
| --- | --- |
| `render_conditions.csv` | The 12 grid conditions and the 6 perturbation conditions |
| `grid_render_metadata.csv` | Per-image parameters, dimensions, and SHA-256 for 12 × 552 images |
| `perturbation_render_metadata.csv` | Same for 6 × 552 images |
| `RENDER_MANIFEST.json` | Renderer version and global settings |

Every digest here corresponds to a PNG shipped under `images/`.

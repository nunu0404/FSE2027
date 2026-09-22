# F6 Difficulty-bin Provenance

## Verdict

Use **study-specific descriptive/legacy bins**. The records do not support saying that the thresholds directly apply Cohen's conventional standardized-effect categories. `|Delta z|` is not Cohen's d, and the easy cutoff 1.0 differs from the common 0.8 convention.

## Definition

For snippet s in benchmark/dataset d:

`z_s = (mean_rating_s - mean_d) / population_SD_d`

where the implementation uses `std(ddof=0)`. Pair difficulty is:

`|Delta z| = |z_i - z_j|`.

This is the absolute difference between two dataset-standardized snippet mean ratings. It is not Cohen's d because it does not divide a raw group-mean difference by a pooled within-group standard deviation and does not use rater-level sampling variance.

Sources:

- `/ANON/experiment_root/experiments/rq0_viability/scripts/prepare_datasets.py`, fields `human_score_z_within_dataset`, `scores.mean()`, and `scores.std(ddof=0)`.
- `/ANON/experiment_root/experiments/rq0_viability/scripts/build_pair_sets.py`, function `difficulty` and fields `score_a`, `score_b`, `abs_diff`.
- `/ANON/experiment_root/experiments/rq0_viability/data/pairs/full_pair_set_manifest.json`.

## Recorded thresholds

- hard: `0.2 <= |Delta z| < 0.5`
- medium: `0.5 <= |Delta z| < 1.0`
- easy: `|Delta z| >= 1.0`
- excluded: `|Delta z| < 0.2` or exact score tie

The pair-construction script predates the full pair manifest (script mtime 2026-06-18 16:24:27 UTC; manifest generated 2026-06-19 08:43:50 UTC), so the thresholds were fixed in code before that 3,000-pair artifact was generated. This is local filesystem provenance, not an externally timestamped preregistration.

No pre-run document found in the searched experiment tree explains why 1.0 was chosen or links the three boundaries to Cohen. A later amendment explicitly says the old cut points are not defensible as established standards and makes continuous `|Delta z|` primary:

`/ANON/experiment_root/results/grounded_protocol_3lang_20260721/GROUNDED_PROTOCOL_AMENDMENT.md`

The later model-battery protocol likewise records `primary_difficulty: continuous abs_z_diff` and `legacy_difficulty_bins: sampling and reproducibility only`:

`/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/config/protocol_preregistered.json`

## Paper-ready text

"We analyze difficulty primarily as the continuous absolute difference between the two within-benchmark standardized mean ratings, |Delta z|. For balanced sampling and legacy reproducibility only, we report study-specific bins: hard [0.2, 0.5), medium [0.5, 1.0), and easy >=1.0; pairs below 0.2 are excluded."

Remove any sentence that says these boundaries directly instantiate Cohen's small/medium/large conventions. A weaker "inspired by standardized-difference magnitudes" claim is also unsupported by the surviving provenance and is unnecessary.


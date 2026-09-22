# Complementarity Mechanism Inventory

Generated at: 2026-07-06T12:42:15.317032+00:00

## Files Found
- rq0_dir: `/ANON/experiment_root/experiments/rq0_viability` (FOUND)
- dataset: `/ANON/experiment_root/experiments/rq0_viability/data/processed/pooled_313_processed.csv` (FOUND)
- source_features: `/ANON/experiment_root/experiments/rq0_viability/data/processed/features_313.csv` (FOUND)
- render_metadata: `/ANON/experiment_root/experiments/rq0_viability/outputs/render_metadata/default_render_metadata.csv` (FOUND)
- rf_pair_level: `/ANON/experiment_root/experiments/rq0_viability/outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv` (FOUND)
- best_vlm_raw: `/ANON/experiment_root/experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__combined_labeled_image_only__promptB__seed42.jsonl` (FOUND)

## Scripts Reused
- Existing feature extractor: `/ANON/experiment_root/experiments/rq0_viability/scripts/extract_features.py`
- Existing RF pair-level output: `/ANON/experiment_root/experiments/rq0_viability/outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv`
- Clean VLM raw output: `/ANON/experiment_root/experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__combined_labeled_image_only__promptB__seed42.jsonl`

## Missing Files
- None

## Regenerated Files
- All files under `/ANON/experiment_root/results/complementarity_mechanism_clean_internvl_combined_image_only_20260706` are newly generated.

## Selected Conditions
- RF prediction file: existing full-pair output filtered to `random_forest_regressor`.
- VLM condition: `OpenGVLab/InternVL3-8B / combined_labeled_image_only / promptB`.

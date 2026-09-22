# Complementarity Mechanism Analysis

## Required inputs
- `/ANON/experiment_root/experiments/rq0_viability/outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv`
- `/ANON/experiment_root/experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__combined_labeled_image_only__promptB__seed42.jsonl`
- `/ANON/experiment_root/experiments/rq0_viability/data/processed/features_313.csv`
- `/ANON/experiment_root/experiments/rq0_viability/outputs/render_metadata/default_render_metadata.csv`

## Command order
Run `python experiments/rq0_viability/scripts/run_complementarity_mechanism_analysis.py --vlm-raw /ANON/experiment_root/experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__combined_labeled_image_only__promptB__seed42.jsonl --output-dir /ANON/experiment_root/results/complementarity_mechanism_clean_qwen_combined_image_only_20260706`.

## Expected runtime
CPU-only; typically under a few minutes because no VLM inference is performed.

## Known limitations
Pair observations share snippets; visual features are proxy measurements; results are correlational.

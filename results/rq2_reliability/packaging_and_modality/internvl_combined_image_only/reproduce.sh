#!/usr/bin/env bash
set -euo pipefail
cd /ANON/experiment_root
python experiments/rq0_viability/scripts/run_complementarity_mechanism_analysis.py --vlm-raw /ANON/experiment_root/experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__combined_labeled_image_only__promptB__seed42.jsonl --output-dir /ANON/experiment_root/results/complementarity_mechanism_clean_internvl_combined_image_only_20260706

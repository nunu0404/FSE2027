#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

cd /ANON/experiment_root/results/text_ablation_20260920

echo "Running OCR condition..."
python3 run_vlm_text_ablation.py --condition ocr

echo "Running Source condition..."
python3 run_vlm_text_ablation.py --condition source

echo "All done!"

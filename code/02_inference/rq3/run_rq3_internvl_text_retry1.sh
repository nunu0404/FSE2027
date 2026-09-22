#!/usr/bin/env bash
set -euo pipefail

ROOT=/ANON/experiment_root
OUT="$ROOT/results/grounded_protocol_3lang_20260721"
PY=/ANON/home/miniconda3/bin/python
RUNNER="$OUT/code/run_rq3_vlm.py"
VALIDATOR="$OUT/code/validate_ga0.py"
RAW="$OUT/rq3/inference/raw"
PILOT="$RAW/OpenGVLab__InternVL3-8B__text_plus_image__promptB__seed42__ga0_retry1_20260721.jsonl"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"

"$PY" "$OUT/code/validate_rq3_gc1.py"
echo "[$(date --iso-8601=seconds)] retry-GA0 start"
"$PY" "$RUNNER" \
  --model OpenGVLab/InternVL3-8B \
  --modality text_plus_image \
  --limit-calls 100 \
  --output-tag ga0_retry1_20260721
"$PY" "$VALIDATOR" --raw "$PILOT"

echo "[$(date --iso-8601=seconds)] full start"
"$PY" "$RUNNER" \
  --model OpenGVLab/InternVL3-8B \
  --modality text_plus_image \
  --output-tag full_20260721
echo "[$(date --iso-8601=seconds)] complete"

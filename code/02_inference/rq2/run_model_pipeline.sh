#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MODEL" >&2
  exit 2
fi

ROOT=/ANON/experiment_root
OUT="$ROOT/results/grounded_protocol_3lang_20260721"
PY=/ANON/home/miniconda3/bin/python
MODEL=$1
SAFE=${MODEL//\//__}
RUNNER="$OUT/code/run_grounded_vlm.py"
VALIDATOR="$OUT/code/validate_ga0.py"
RAW_DIR="$OUT/inference/grid/raw"
GA0_TAG=ga0b_20260721
FULL_TAG=full_20260721

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"

echo "[$(date --iso-8601=seconds)] GA0 start: $MODEL"
"$PY" "$RUNNER" --experiment grid --model "$MODEL" --limit-calls 100 --output-tag "$GA0_TAG"
"$PY" "$VALIDATOR" --raw "$RAW_DIR/${SAFE}__image_only__promptB__seed42__${GA0_TAG}.jsonl"

echo "[$(date --iso-8601=seconds)] A grid start: $MODEL"
"$PY" "$RUNNER" --experiment grid --model "$MODEL" --output-tag "$FULL_TAG"

echo "[$(date --iso-8601=seconds)] B perturbation start: $MODEL"
"$PY" "$RUNNER" --experiment perturbation --model "$MODEL" --output-tag "$FULL_TAG"

echo "[$(date --iso-8601=seconds)] pipeline complete: $MODEL"

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
RUNNER="$OUT/code/run_rq3_vlm.py"
VALIDATOR="$OUT/code/validate_ga0.py"
RAW_DIR="$OUT/rq3/inference/raw"
PILOT_TAG=ga0_20260721
FULL_TAG=full_20260721

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"

"$PY" "$OUT/code/validate_rq3_gc1.py"
for MODALITY in image_only text_plus_image; do
  echo "[$(date --iso-8601=seconds)] RQ3 GA0 start: $MODEL / $MODALITY"
  "$PY" "$RUNNER" --model "$MODEL" --modality "$MODALITY" --limit-calls 100 --output-tag "$PILOT_TAG"
  "$PY" "$VALIDATOR" --raw "$RAW_DIR/${SAFE}__${MODALITY}__promptB__seed42__${PILOT_TAG}.jsonl"

  echo "[$(date --iso-8601=seconds)] RQ3 full start: $MODEL / $MODALITY"
  "$PY" "$RUNNER" --model "$MODEL" --modality "$MODALITY" --output-tag "$FULL_TAG"
done
echo "[$(date --iso-8601=seconds)] RQ3 pipeline complete: $MODEL"

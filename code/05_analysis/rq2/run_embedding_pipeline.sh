#!/usr/bin/env bash
set -euo pipefail

ROOT=/ANON/experiment_root
OUT="$ROOT/results/grounded_protocol_3lang_20260721"
PY=/ANON/home/miniconda3/bin/python
RUN="$OUT/code/run_embedding_extraction.py"
VALIDATE="$OUT/code/validate_embedding_run.py"

GPU_INDEX=${GPU_INDEX:-0}
MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"

while true; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_INDEX" | tr -d ' ')
  if (( free_mib >= MIN_FREE_MIB )); then
    echo "[$(date --iso-8601=seconds)] GPU${GPU_INDEX} memory gate pass: ${free_mib} MiB free"
    break
  fi
  echo "[$(date --iso-8601=seconds)] waiting for GPU${GPU_INDEX}: ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB"
  sleep 300
done

for model in Qwen/Qwen2.5-VL-7B-Instruct OpenGVLab/InternVL3-8B; do
  safe=${model//\//__}
  echo "[$(date --iso-8601=seconds)] embedding pilot start: $model"
  "$PY" "$RUN" --model "$model" --limit 17 --output-tag pilot17b_20260722
  "$PY" "$VALIDATE" --manifest "$OUT/embedding/raw/${safe}__vision__pilot17b_20260722.manifest.json" --expected 17
  echo "[$(date --iso-8601=seconds)] embedding full start: $model"
  "$PY" "$RUN" --model "$model" --output-tag full_20260722
  "$PY" "$VALIDATE" --manifest "$OUT/embedding/raw/${safe}__vision__full_20260722.manifest.json" --expected 9384
done
echo "[$(date --iso-8601=seconds)] embedding pipeline complete"

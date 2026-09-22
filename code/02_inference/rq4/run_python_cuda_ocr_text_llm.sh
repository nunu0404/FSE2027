#!/usr/bin/env bash
set -euo pipefail

ROOT="/ANON/experiment_root"
PY="/ANON/home/miniconda3/bin/python"
RESULTS="results/python_cuda_missing_experiments_20260716"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

cd "$ROOT"
mkdir -p "$RESULTS/ocr_text_llm_raw" "$RESULTS/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

"$PY" experiments/rq0_viability/scripts/manage_python_cuda_ocr_text_llm.py prepare

run_condition() {
  local name="$1"
  local dataset="$2"
  "$PY" experiments/rq0_viability/scripts/run_judge_pairs.py \
    --model "$MODEL" \
    --model-kind text_llm \
    --input-setting text_only \
    --prompt-variant B \
    --pairs results/python_cuda_vlm_main_20260715/data/pairs_seed42_clean.csv \
    --dataset "$dataset" \
    --render-metadata results/python_cuda_vlm_main_20260715/render_metadata/default_render_metadata.csv \
    --output-dir "$RESULTS/ocr_text_llm_raw" \
    --run-name "$name" \
    --limit-pairs 6000 \
    --resume \
    --dtype bf16 \
    --device-map auto \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-new-tokens 24 \
    --seed 42
}

run_condition source_text_llm "$RESULTS/ocr_text_llm_inputs/source_text.csv"
run_condition ocr_text_llm_rapidocr "$RESULTS/ocr_text_llm_inputs/rapidocr_text.csv"
run_condition ocr_text_llm_easyocr "$RESULTS/ocr_text_llm_inputs/easyocr_text.csv"
run_condition ocr_text_llm_easyocr_preprocessed "$RESULTS/ocr_text_llm_inputs/easyocr_preprocessed_text.csv"

"$PY" experiments/rq0_viability/scripts/manage_python_cuda_ocr_text_llm.py finalize

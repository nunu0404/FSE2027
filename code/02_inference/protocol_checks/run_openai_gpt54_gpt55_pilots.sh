#!/usr/bin/env bash
set -euo pipefail

ROOT="/ANON/experiment_root"
PY="/ANON/home/miniconda3/bin/python"
OUT="results/openai_vlm_pilot_20260708"
LOG_DIR="${OUT}/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set in this shell/tmux environment" >&2
  exit 1
fi

echo "[openai_gpt54_gpt55] started_at=$(date -Is)"
echo "[openai_gpt54_gpt55] output_dir=${OUT}"

run_model() {
  local model="$1"
  local input_cost="$2"
  local output_cost="$3"
  echo "[openai_gpt54_gpt55] model=${model} started_at=$(date -Is)"
  "${PY}" experiments/rq0_viability/scripts/run_openai_vlm_pilot.py \
    --model "${model}" \
    --output-dir "${OUT}" \
    --pairs-per-difficulty 100 \
    --detail high \
    --resume \
    --input-cost-per-mtok "${input_cost}" \
    --output-cost-per-mtok "${output_cost}" \
    2>&1 | tee "${LOG_DIR}/${model//\//__}.log"
  echo "[openai_gpt54_gpt55] model=${model} finished_at=$(date -Is)"
}

run_model "gpt-5.4" "5.00" "30.00"
run_model "gpt-5.5" "5.00" "30.00"

echo "[openai_gpt54_gpt55] finished_at=$(date -Is)"

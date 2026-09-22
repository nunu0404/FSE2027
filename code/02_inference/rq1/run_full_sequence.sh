#!/usr/bin/env bash
set -euo pipefail

ROOT=/ANON/experiment_root
OUT="$ROOT/results/rq1_model_battery_3lang_20260723"
BASE_PY=/ANON/home/miniconda3/bin/python
PHI_PY=/ANON/scratch_rq1/envs/phi4mm/bin/python
RUNNER="$OUT/code/run_rq1_vlm.py"

export CUDA_VISIBLE_DEVICES=0
export RQ1_GPU_NAME="NVIDIA RTX PRO 6000 Blackwell Server Edition"

cd "$ROOT"
mkdir -p "$OUT/logs/full"

run_model() {
  local model=$1
  local interpreter=$2
  local summary="$OUT/inference/ga0/$model/summary.json"
  "$BASE_PY" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["gate_pass"] is True' \
    "$summary"
  if [[ "$model" == qwen || "$model" == internvl ]]; then
    env -u HF_HOME "$interpreter" "$RUNNER" --model "$model" --phase full --resume \
      2>&1 | tee "$OUT/logs/full/$model.log"
  else
    HF_HOME=/ANON/scratch_rq1/hf \
      "$interpreter" "$RUNNER" --model "$model" --phase full --resume \
      2>&1 | tee "$OUT/logs/full/$model.log"
  fi
}

run_model qwen "$BASE_PY"
run_model internvl "$BASE_PY"
run_model gemma "$BASE_PY"
run_model ministral "$BASE_PY"
run_model phi "$PHI_PY"

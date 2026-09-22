#!/usr/bin/env bash
set -uo pipefail

ROOT=/ANON/experiment_root
OUT="$ROOT/results/rq1_model_battery_3lang_20260723"
BASE_PY=/ANON/home/miniconda3/bin/python
PHI_PY=/ANON/scratch_rq1/envs/phi4mm/bin/python
RUNNER="$OUT/code/run_rq1_vlm.py"

export CUDA_VISIBLE_DEVICES=0
export RQ1_GPU_NAME="NVIDIA RTX PRO 6000 Blackwell Server Edition"

cd "$ROOT"
mkdir -p "$OUT/logs/full_parallel"

run_model() {
  local model=$1
  local interpreter=$2
  local group=$3
  local summary="$OUT/inference/ga0/$model/summary.json"
  "$BASE_PY" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["gate_pass"] is True' \
    "$summary" || return $?
  if [[ "$model" == qwen || "$model" == internvl ]]; then
    env -u HF_HOME RQ1_CONCURRENCY_GROUP="$group" \
      "$interpreter" "$RUNNER" --model "$model" --phase full --resume \
      > >(tee "$OUT/logs/full_parallel/$model.log") 2>&1
  else
    HF_HOME=/ANON/scratch_rq1/hf RQ1_CONCURRENCY_GROUP="$group" \
      "$interpreter" "$RUNNER" --model "$model" --phase full --resume \
      > >(tee "$OUT/logs/full_parallel/$model.log") 2>&1
  fi
}

run_pair() {
  local model_a=$1
  local python_a=$2
  local model_b=$3
  local python_b=$4
  local group=$5
  run_model "$model_a" "$python_a" "$group" &
  local pid_a=$!
  run_model "$model_b" "$python_b" "$group" &
  local pid_b=$!
  local status_a=0
  local status_b=0
  wait "$pid_a" || status_a=$?
  wait "$pid_b" || status_b=$?
  if (( status_a != 0 || status_b != 0 )); then
    echo "parallel group $group failed: $model_a=$status_a $model_b=$status_b" >&2
    return 1
  fi
}

run_pair qwen "$BASE_PY" internvl "$BASE_PY" pair1_qwen_internvl || exit 1
run_pair gemma "$BASE_PY" ministral "$BASE_PY" pair2_gemma_ministral || exit 1
run_model phi "$PHI_PY" group3_phi_alone || exit 1

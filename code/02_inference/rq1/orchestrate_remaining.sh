#!/usr/bin/env bash
set -euo pipefail

ROOT=/ANON/experiment_root
OUT="$ROOT/results/latest_vlm_extension_20260830"
HF_HOME=/ANON/scratch_rq1/hf
PYTHON=/ANON/home/miniconda3/bin/python

log() {
  printf '[%s KST] %s\n' "$(TZ=Asia/Seoul date '+%F %T')" "$*"
}

gate_pass() {
  "$PYTHON" - "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and json.loads(path.read_text()).get("gate_pass") else 1)
PY
}

wait_for_gate() {
  local session=$1
  local validation=$2
  while true; do
    if gate_pass "$validation"; then
      log "PASS $validation"
      return 0
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      log "FAIL: $session ended without a passing validation: $validation"
      return 1
    fi
    sleep 20
  done
}

launch_robustness() {
  local model=$1
  local gpu=$2
  local session="latestvlm_robust_${model}_gpu${gpu}"
  local logfile="$OUT/logs/robustness_${model}_gpu${gpu}.log"
  local validation="$OUT/inference/robustness/${model}/validation.json"
  if gate_pass "$validation"; then
    log "robustness already has a passing validation: $model"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    log "session already exists: $session"
    return
  fi
  tmux new-session -d -s "$session" \
    "cd '$OUT' && CUDA_VISIBLE_DEVICES='$gpu' HF_HOME='$HF_HOME' '$PYTHON' code/run_robustness.py --model '$model' --resume > '$logfile' 2>&1"
  log "launched $session with CUDA_VISIBLE_DEVICES=$gpu"
}

log "waiting for all primary inference integrity gates"
wait_for_gate latestvlm_full_qwen3_gpu0 \
  "$OUT/inference/full/qwen3/validation.json"
wait_for_gate latestvlm_full_internvl35_gpu1 \
  "$OUT/inference/full/internvl3_5/validation.json"
wait_for_gate latestvlm_gemma4_gpu0_pipeline \
  "$OUT/inference/full/gemma4/validation.json"

log "all primary gates passed; running primary analysis"
cd "$OUT"
"$PYTHON" code/amend_primary_generation_metadata.py \
  > "$OUT/logs/amend_primary_generation_metadata.log" 2>&1
"$PYTHON" code/analyze_primary.py > "$OUT/logs/analyze_primary.log" 2>&1
log "primary analysis complete"

launch_robustness qwen3 0
launch_robustness internvl3_5 1
wait_for_gate latestvlm_robust_qwen3_gpu0 \
  "$OUT/inference/robustness/qwen3/validation.json"
wait_for_gate latestvlm_robust_internvl3_5_gpu1 \
  "$OUT/inference/robustness/internvl3_5/validation.json"

log "all robustness gates passed; running paired robustness analysis"
"$PYTHON" code/analyze_robustness.py > "$OUT/logs/analyze_robustness.log" 2>&1
log "robustness analysis complete"

DEPLOY_SESSION=latestvlm_deploy_qwen3_gpu0
DEPLOY_LOG="$OUT/logs/deployment_qwen3_gpu0.log"
DEPLOY_VALIDATION="$OUT/inference/deployment/qwen3_direct_visual/validation.json"
if gate_pass "$DEPLOY_VALIDATION"; then
  log "deployment already has a passing validation"
elif ! tmux has-session -t "$DEPLOY_SESSION" 2>/dev/null; then
  tmux new-session -d -s "$DEPLOY_SESSION" \
    "cd '$OUT' && CUDA_VISIBLE_DEVICES=0 HF_HOME='$HF_HOME' '$PYTHON' code/run_deployment.py --resume > '$DEPLOY_LOG' 2>&1"
  log "launched $DEPLOY_SESSION with CUDA_VISIBLE_DEVICES=0"
fi
wait_for_gate "$DEPLOY_SESSION" \
  "$DEPLOY_VALIDATION"

log "deployment gate passed; running deployment analysis"
"$PYTHON" code/analyze_deployment.py > "$OUT/logs/analyze_deployment.log" 2>&1
log "deployment analysis complete"

"$PYTHON" code/build_final_provenance.py > "$OUT/logs/build_final_provenance.log" 2>&1
"$PYTHON" code/generate_final_report.py > "$OUT/logs/generate_final_report.log" 2>&1
log "all requested stages complete; final report generated"

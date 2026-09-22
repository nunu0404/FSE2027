#!/usr/bin/env bash
set -euo pipefail

ROOT="/ANON/experiment_root"
cd "$ROOT"

PY_GEMMA4="/ANON/scratch_rq1/envs/latest_gemma4_tf5161/bin/python"
PY_ANALYSIS="/usr/bin/python"
LOGFILE="rq2_eval_reduced/logs/gemma4_gpu0.log"
mkdir -p rq2_eval_reduced/logs

export LD_LIBRARY_PATH="/ANON/home/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

echo "[PIPELINE] Starting True Gemma4-12B (707f0a3) RQ2 Pipeline on GPU 0..." | tee -a "$LOGFILE"
echo "[PIPELINE] Start Time: $(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOGFILE"

# Step 1: Run Inference (42,000 calls)
echo "[PIPELINE] Executing inference..." | tee -a "$LOGFILE"
"$PY_GEMMA4" rq2_eval_reduced/scripts/run_gemma4_reduced.py --device 0 2>&1 | tee -a "$LOGFILE"

# Step 2: Run Statistical Analysis & Handoff Bundle Packaging
echo "[PIPELINE] Inference complete! Running analysis & packaging..." | tee -a "$LOGFILE"
"$PY_ANALYSIS" rq2_eval_reduced/scripts/analyze_gemma4_reduced.py 2>&1 | tee -a "$LOGFILE"

echo "[PIPELINE] Gemma4 Pipeline Finished Successfully at: $(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOGFILE"

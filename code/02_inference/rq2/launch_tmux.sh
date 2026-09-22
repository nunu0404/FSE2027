#!/usr/bin/env bash
set -euo pipefail

ROOT="/ANON/experiment_root"
cd "$ROOT"

echo "[LAUNCHER] Starting RQ2 Reduced Grid Parallel Execution..."
echo "[LAUNCHER] GPU 0: Gemma-3-12b-it (42,000 calls) + Qwen Anchor (6,000 calls)"
echo "[LAUNCHER] GPU 1: InternVL3.5-8B-HF (42,000 calls)"

# Kill old sessions if they exist
tmux kill-session -t rq2_gemma_gpu0 2>/dev/null || true
tmux kill-session -t rq2_internvl_gpu1 2>/dev/null || true

# Session 1: GPU 0 (Gemma)
tmux new-session -d -s rq2_gemma_gpu0 "cd $ROOT && CUDA_VISIBLE_DEVICES=0 /usr/bin/python rq2_eval_reduced/scripts/run_gemma_reduced.py 2>&1 | tee rq2_eval_reduced/logs/gemma_gpu0.log"
echo "[LAUNCHER] Created tmux session 'rq2_gemma_gpu0' on GPU 0. Log: rq2_eval_reduced/logs/gemma_gpu0.log"

# Session 2: GPU 1 (InternVL3.5)
tmux new-session -d -s rq2_internvl_gpu1 "cd $ROOT && CUDA_VISIBLE_DEVICES=1 /usr/bin/python rq2_eval_reduced/scripts/run_internvl_reduced.py 2>&1 | tee rq2_eval_reduced/logs/internvl_gpu1.log"
echo "[LAUNCHER] Created tmux session 'rq2_internvl_gpu1' on GPU 1. Log: rq2_eval_reduced/logs/internvl_gpu1.log"

echo "[LAUNCHER] Both sessions launched successfully!"
tmux list-sessions

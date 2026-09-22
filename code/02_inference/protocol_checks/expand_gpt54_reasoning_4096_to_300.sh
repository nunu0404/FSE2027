#!/usr/bin/env bash
set -euo pipefail

cd /ANON/experiment_root

PYTHON=/ANON/home/miniconda3/bin/python
RUNNER=experiments/rq0_viability/scripts/run_openai_vlm_pilot.py
OUT=results/gpt54_reasoning_budget_pilot_90_20260709
MODEL=gpt-5.4-2026-03-05

mkdir -p "$OUT/logs"

for effort in low high; do
  budget=4096
  run_name="gpt-5.4-2026-03-05__reasoning_${effort}__mot_${budget}"
  printf '[EXPAND START] effort=%s budget=%s target=300 %s\n' \
    "$effort" "$budget" "$(date -u +%FT%TZ)" | tee -a "$OUT/logs/expand_wrapper.log"
  "$PYTHON" "$RUNNER" \
    --model "$MODEL" \
    --output-dir "$OUT" \
    --run-name "$run_name" \
    --conditions image_only text_plus_image \
    --pairs-per-difficulty 100 \
    --detail high \
    --max-output-tokens "$budget" \
    --reasoning-effort "$effort" \
    --request-timeout 180 \
    --timeout-retries 6 \
    --input-cost-per-mtok 2.50 \
    --output-cost-per-mtok 15.00 \
    --resume 2>&1 | tee "$OUT/logs/${run_name}__expand300.log"
  printf '[EXPAND DONE] effort=%s budget=%s target=300 %s\n' \
    "$effort" "$budget" "$(date -u +%FT%TZ)" | tee -a "$OUT/logs/expand_wrapper.log"
done


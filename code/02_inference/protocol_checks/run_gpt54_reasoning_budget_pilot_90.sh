#!/usr/bin/env bash
set -euo pipefail

cd /ANON/experiment_root

PYTHON=/ANON/home/miniconda3/bin/python
RUNNER=experiments/rq0_viability/scripts/run_openai_vlm_pilot.py
OUT=results/gpt54_reasoning_budget_pilot_90_20260709
MODEL=gpt-5.4-2026-03-05

mkdir -p "$OUT/logs"

for effort in low high; do
  for budget in 256 4096; do
    run_name="gpt-5.4-2026-03-05__reasoning_${effort}__mot_${budget}"
    printf '[START] effort=%s budget=%s %s\n' "$effort" "$budget" "$(date -u +%FT%TZ)" \
      | tee -a "$OUT/logs/wrapper.log"
    "$PYTHON" "$RUNNER" \
      --model "$MODEL" \
      --output-dir "$OUT" \
      --run-name "$run_name" \
      --conditions image_only text_plus_image \
      --pairs-per-difficulty 30 \
      --detail high \
      --max-output-tokens "$budget" \
      --reasoning-effort "$effort" \
      --request-timeout 180 \
      --timeout-retries 6 \
      --input-cost-per-mtok 2.50 \
      --output-cost-per-mtok 15.00 \
      --resume 2>&1 | tee "$OUT/logs/${run_name}.log"
    printf '[DONE] effort=%s budget=%s %s\n' "$effort" "$budget" "$(date -u +%FT%TZ)" \
      | tee -a "$OUT/logs/wrapper.log"
  done
done

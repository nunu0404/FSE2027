#!/usr/bin/env bash
set -u

OUT=/ANON/experiment_root/results/rq1_model_battery_3lang_20260723
printf '%-11s %10s %10s\n' MODEL CALLS PERCENT
for model in qwen internvl gemma ministral phi; do
  raw="$OUT/inference/full/$model/raw.jsonl"
  calls=0
  [[ -f "$raw" ]] && calls=$(wc -l < "$raw")
  percent=$(awk -v n="$calls" 'BEGIN { printf "%.2f%%", n/180 }')
  printf '%-11s %10d %10s\n' "$model" "$calls" "$percent"
done
echo
tmux list-sessions 2>/dev/null | grep -E \
  'rq1_full_(sequence|parallel_pairs)' || true
echo
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
echo
for model in qwen internvl gemma ministral phi; do
  log="$OUT/logs/full/$model.log"
  [[ -f "$log" ]] && tail -n 1 "$log"
done
true

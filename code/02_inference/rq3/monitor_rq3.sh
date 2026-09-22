#!/usr/bin/env bash
set -u

ROOT=/ANON/experiment_root
OUT="$ROOT/results/grounded_protocol_3lang_20260721"
RAW="$OUT/rq3/inference/raw"

nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader
echo
tmux ls 2>/dev/null | grep -E 'grounded_(rq3|qwen|internvl)' || true

for model in Qwen__Qwen2.5-VL-7B-Instruct OpenGVLab__InternVL3-8B; do
  echo
  echo "[$model]"
  for modality in image_only text_plus_image; do
    pilot="$RAW/${model}__${modality}__promptB__seed42__ga0_20260721.jsonl"
    gate="$RAW/${model}__${modality}__promptB__seed42__ga0_20260721.ga0.json"
    full="$RAW/${model}__${modality}__promptB__seed42__full_20260721.jsonl"
    p=0; f=0; g=pending
    [[ -f "$pilot" ]] && p=$(wc -l < "$pilot")
    [[ -f "$full" ]] && f=$(wc -l < "$full")
    if [[ -f "$gate" ]]; then
      g=$(/ANON/home/miniconda3/bin/python -c "import json; print('PASS' if json.load(open('$gate'))['gate_pass'] else 'FAIL')")
    fi
    printf '  %-16s GA0 %3d/100 gate=%-7s full %4d/3600 (%6.2f%%)\n' \
      "$modality" "$p" "$g" "$f" "$(awk -v n="$f" 'BEGIN {print n*100/3600}')"
  done
done

echo
echo "--- current RQ3 log ---"
if [[ -f "$OUT/logs/rq3/internvl_pipeline.log" && -s "$OUT/logs/rq3/internvl_pipeline.log" ]]; then
  tail -n 8 "$OUT/logs/rq3/internvl_pipeline.log"
else
  tail -n 8 "$OUT/logs/rq3/qwen_pipeline.log" 2>/dev/null || true
fi

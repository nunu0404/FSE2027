#!/usr/bin/env bash
set -euo pipefail

ROOT="/ANON/experiment_root"
PY="/ANON/home/miniconda3/bin/python"
OUT="results/openai_vlm_pilot_20260708_gpt55_fixed"
LOG_DIR="${OUT}/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set in this shell/tmux environment" >&2
  exit 1
fi

echo "[gpt55_fixed] started_at=$(date -Is)"
echo "[gpt55_fixed] output_dir=${OUT}"

rm -f "${OUT}/gpt-5.5__pilot_raw.jsonl" \
      "${OUT}/gpt-5.5__pilot_summary.csv" \
      "${OUT}/gpt-5.5__pilot_summary.md" \
      "${OUT}/gpt-5.5__pilot_usage.json"

run_once() {
  local pairs_per_difficulty="$1"
  local tag="$2"
  local timeout_retries="$3"
  echo "[gpt55_fixed] ${tag} pairs_per_difficulty=${pairs_per_difficulty} started_at=$(date -Is)"
  "${PY}" experiments/rq0_viability/scripts/run_openai_vlm_pilot.py \
    --model "gpt-5.5" \
    --output-dir "${OUT}" \
    --pairs-per-difficulty "${pairs_per_difficulty}" \
    --detail high \
    --max-output-tokens 256 \
    --reasoning-effort low \
    --request-timeout 90 \
    --timeout-retries "${timeout_retries}" \
    --input-cost-per-mtok 5.00 \
    --output-cost-per-mtok 30.00 \
    2>&1 | tee "${LOG_DIR}/gpt-5.5.${tag}.log"
  echo "[gpt55_fixed] ${tag} finished_at=$(date -Is)"
}

# Do a tiny non-resume smoke first. If this still parse-fails, stop before the full run.
run_once 3 "smoke" 1

"${PY}" - <<'PY'
import json
from pathlib import Path
p = Path("results/openai_vlm_pilot_20260708_gpt55_fixed/gpt-5.5__pilot_raw.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("smoke produced no rows")
parse_fail_rate = sum(r.get("parse_failures", 0) for r in rows) / (2 * len(rows))
valid_rate = sum(bool(r.get("is_valid_strict_swap")) for r in rows) / len(rows)
print(f"[gpt55_fixed] smoke_rows={len(rows)} parse_fail_rate={parse_fail_rate:.4f} valid_rate={valid_rate:.4f}")
if parse_fail_rate > 0.25 or valid_rate == 0:
    raise SystemExit("smoke failed: parse/valid pattern still abnormal; not starting full run")
PY

# Replace the smoke file with a full clean run after the smoke gate passes.
rm -f "${OUT}/gpt-5.5__pilot_raw.jsonl" \
      "${OUT}/gpt-5.5__pilot_summary.csv" \
      "${OUT}/gpt-5.5__pilot_summary.md" \
      "${OUT}/gpt-5.5__pilot_usage.json"

run_once 100 "full" 3

echo "[gpt55_fixed] finished_at=$(date -Is)"

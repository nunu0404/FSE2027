#!/usr/bin/env bash
set -euo pipefail

ROOT="/ANON/experiment_root"
PY="/ANON/home/miniconda3/bin/python"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
OUT="results/screenshot_only_ocr/ocr_llm_raw"
LOG_DIR="results/screenshot_only_ocr/logs"

cd "$ROOT"
mkdir -p "$OUT" "$LOG_DIR" "results/screenshot_only_ocr/llm_inputs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo "[ocr_text_llm] started_at=$(date -Is)"
echo "[ocr_text_llm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "[ocr_text_llm] downloading/checking ${MODEL}"
huggingface-cli download "${MODEL}" --local-dir-use-symlinks False

echo "[ocr_text_llm] preparing OCR dataset"
"${PY}" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/ANON/experiment_root")
dataset_path = root / "experiments/rq0_viability/data/processed/pooled_313_processed.csv"
ocr_path = root / "results/screenshot_only_ocr/ocr_outputs/easyocr/snippet_ocr.csv"
out_path = root / "results/screenshot_only_ocr/llm_inputs/pooled_313_easyocr_text.csv"

dataset = pd.read_csv(dataset_path)
ocr = pd.read_csv(ocr_path).rename(columns={"snippet_id": "rq0_id"})
merged = dataset.merge(ocr[["rq0_id", "ocr_text"]], on="rq0_id", how="left")
if merged["ocr_text"].isna().any():
    raise SystemExit("Missing OCR text for some snippets")
merged["raw_code"] = merged["ocr_text"].fillna("").astype(str)
merged = merged.drop(columns=["ocr_text"])
out_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_path, index=False)
print({"rows": len(merged), "output": str(out_path)})
PY

echo "[ocr_text_llm] source text run"
"${PY}" experiments/rq0_viability/scripts/run_judge_pairs.py \
  --model "${MODEL}" \
  --model-kind text_llm \
  --input-setting text_only \
  --prompt-variant B \
  --pairs experiments/rq0_viability/data/pairs/full_pair_set_rq0.csv \
  --dataset experiments/rq0_viability/data/processed/pooled_313_processed.csv \
  --output-dir "${OUT}" \
  --run-name source_text_llm \
  --limit-pairs 3000 \
  --resume \
  --dtype bf16 \
  --device-map auto \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-new-tokens 24 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/source_text_llm_qwen25_coder.log"

echo "[ocr_text_llm] OCR text run"
"${PY}" experiments/rq0_viability/scripts/run_judge_pairs.py \
  --model "${MODEL}" \
  --model-kind text_llm \
  --input-setting text_only \
  --prompt-variant B \
  --pairs experiments/rq0_viability/data/pairs/full_pair_set_rq0.csv \
  --dataset results/screenshot_only_ocr/llm_inputs/pooled_313_easyocr_text.csv \
  --output-dir "${OUT}" \
  --run-name ocr_text_llm \
  --limit-pairs 3000 \
  --resume \
  --dtype bf16 \
  --device-map auto \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-new-tokens 24 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/ocr_text_llm_qwen25_coder.log"

echo "[ocr_text_llm] finalizing"
"${PY}" experiments/rq0_viability/scripts/finalize_ocr_text_llm_baseline.py 2>&1 | tee "${LOG_DIR}/finalize_ocr_text_llm.log"

echo "[ocr_text_llm] finished_at=$(date -Is)"

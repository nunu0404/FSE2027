#!/usr/bin/env bash
set -euo pipefail

ROOT="/ANON/experiment_root"
PY="/ANON/home/miniconda3/bin/python"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
RESULTS="results/screenshot_only_ocr_clean_20260707"
RAW="${RESULTS}/ocr_text_llm_raw"
INPUTS="${RESULTS}/llm_inputs"
LOG_DIR="${RESULTS}/logs"

cd "$ROOT"
mkdir -p "$RAW" "$INPUTS" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo "[clean_ocr_text_llm] started_at=$(date -Is)"
echo "[clean_ocr_text_llm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "[clean_ocr_text_llm] downloading/checking ${MODEL}"
huggingface-cli download "${MODEL}"

echo "[clean_ocr_text_llm] preparing OCR text datasets"
"${PY}" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/ANON/experiment_root")
dataset_path = root / "experiments/rq0_viability/data/processed/pooled_313_processed.csv"
ocr_path = root / "results/screenshot_only_ocr_clean_20260707/clean_ocr_text_by_snippet_all_engines.csv"
out_dir = root / "results/screenshot_only_ocr_clean_20260707/llm_inputs"
dataset = pd.read_csv(dataset_path)
ocr = pd.read_csv(ocr_path)
out_dir.mkdir(parents=True, exist_ok=True)
for engine in ["easyocr", "rapidocr", "easyocr_preprocessed"]:
    sub = ocr[ocr["ocr_engine"].eq(engine)][["snippet_id", "ocr_text"]].rename(columns={"snippet_id": "rq0_id"})
    merged = dataset.merge(sub, on="rq0_id", how="left")
    if merged["ocr_text"].isna().any():
        missing = merged.loc[merged["ocr_text"].isna(), "rq0_id"].head().tolist()
        raise SystemExit(f"Missing OCR text for {engine}: {missing}")
    merged["raw_code"] = merged["ocr_text"].fillna("").astype(str)
    merged = merged.drop(columns=["ocr_text"])
    out = out_dir / f"pooled_313_{engine}_text.csv"
    merged.to_csv(out, index=False)
    print({"engine": engine, "rows": len(merged), "output": str(out)})
PY

run_text_judge() {
  local run_name="$1"
  local dataset="$2"
  local log_name="$3"
  echo "[clean_ocr_text_llm] run=${run_name} dataset=${dataset}"
  "${PY}" experiments/rq0_viability/scripts/run_judge_pairs.py \
    --model "${MODEL}" \
    --model-kind text_llm \
    --input-setting text_only \
    --prompt-variant B \
    --pairs experiments/rq0_viability/data/pairs/full_pair_set_rq0.csv \
    --dataset "${dataset}" \
    --output-dir "${RAW}" \
    --run-name "${run_name}" \
    --limit-pairs 3000 \
    --resume \
    --dtype bf16 \
    --device-map auto \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-new-tokens 24 \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/${log_name}.log"
}

run_text_judge "source_text_llm" "experiments/rq0_viability/data/processed/pooled_313_processed.csv" "clean_source_text_llm_qwen25_coder"
run_text_judge "ocr_text_llm_easyocr" "${INPUTS}/pooled_313_easyocr_text.csv" "clean_ocr_text_llm_easyocr_qwen25_coder"
run_text_judge "ocr_text_llm_rapidocr" "${INPUTS}/pooled_313_rapidocr_text.csv" "clean_ocr_text_llm_rapidocr_qwen25_coder"
run_text_judge "ocr_text_llm_easyocr_preprocessed" "${INPUTS}/pooled_313_easyocr_preprocessed_text.csv" "clean_ocr_text_llm_easyocr_preprocessed_qwen25_coder"

echo "[clean_ocr_text_llm] finalizing"
"${PY}" experiments/rq0_viability/scripts/finalize_clean_ocr_text_llm.py 2>&1 | tee "${LOG_DIR}/finalize_clean_ocr_text_llm.log"

echo "[clean_ocr_text_llm] finished_at=$(date -Is)"

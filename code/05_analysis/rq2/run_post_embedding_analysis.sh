#!/usr/bin/env bash
set -euo pipefail

ROOT=/ANON/experiment_root
OUT="$ROOT/results/grounded_protocol_3lang_20260721"
PY=/ANON/home/miniconda3/bin/python

cd "$ROOT"
while tmux has-session -t grounded_embedding_gpu1_20260722 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] waiting for embedding extraction"
  sleep 30
done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("results/grounded_protocol_3lang_20260721/embedding/raw")
paths = sorted(root.glob("*__vision__full_20260722.manifest.json"))
assert len(paths) == 2, paths
for path in paths:
    manifest = json.loads(path.read_text())
    assert manifest.get("status") == "complete", (path, manifest.get("status"))
    assert manifest.get("completed_images") == 9384, (path, manifest.get("completed_images"))
print("embedding full manifests: PASS")
PY

"$PY" "$OUT/code/analyze_embedding_stability.py"
"$PY" "$OUT/code/build_integrated_final_report.py"
echo "[$(date --iso-8601=seconds)] post-embedding analysis complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/ANON/experiment_root
OUT="$ROOT/results/fse2027_final_verification"
cd "$ROOT"

# F1, F2, F4: deterministic extraction from frozen CSV/JSON/JSONL assets.
python "$OUT/code/build_static_audits.py"

# F3: 10,000 pair-cluster and crossed-endpoint bootstrap draws per group/metric.
python "$OUT/code/recompute_logit_cluster_statistics.py" \
  --resamples 10000 \
  --seed 20260820

# F5 sensitivity: five endpoint-disjoint folds, seed 42. Crossing pairs are excluded.
python "$OUT/code/recompute_mechanism_snippet_disjoint.py"

# Validate generated structured files.
python -m json.tool "$OUT/qwen35_run_metadata.json" >/dev/null
python -m json.tool "$OUT/rendering_contrast_definition.json" >/dev/null
python - <<'PY'
from pathlib import Path
import pandas as pd

out = Path("/ANON/experiment_root/results/fse2027_final_verification")
expected = {
    "ocr_loss_audit.csv": 3,
    "logit_cluster_statistics.csv": 24,
    "original_logit_mannwhitney_audit.csv": 4,
    "mechanism_snippet_disjoint_auc.csv": 22,
}
for name, rows in expected.items():
    actual = len(pd.read_csv(out / name))
    if actual != rows:
        raise SystemExit(f"{name}: expected {rows} rows, found {actual}")
print("structured-output validation passed")
PY

# Hash every generated artifact except the hash list itself.
find "$OUT" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUT/SHA256SUMS"
sha256sum -c "$OUT/SHA256SUMS"

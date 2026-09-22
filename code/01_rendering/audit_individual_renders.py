#!/usr/bin/env python3
"""Re-render every source snippet and compare all pixels with prepared PNGs."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageChops
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_code_images import wrap_code  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--render-metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    data = pd.read_csv(repo / args.dataset).set_index("rq0_id")
    metadata = pd.read_csv(repo / args.render_metadata)
    rendering = yaml.safe_load((repo / args.config).read_text(encoding="utf-8"))["rendering"]["default"]
    font_path = repo / rendering["font_name"]
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    checked = 0

    if len(metadata) != len(data) or metadata["rq0_id"].nunique() != len(data):
        failures.append({"rq0_id": "__metadata__", "error": "dataset/render metadata cardinality mismatch"})

    for row in metadata.itertuples(index=False):
        try:
            snippet = data.loc[row.rq0_id]
            wrapped = wrap_code(snippet.raw_code, int(rendering["wrap_column"]))
            try:
                lexer = get_lexer_by_name(str(snippet.language))
            except ClassNotFound:
                lexer = TextLexer()
            formatter = ImageFormatter(
                font_name=str(font_path),
                font_size=int(rendering["font_size"]),
                line_numbers=bool(rendering["line_numbers"]),
                style=str(rendering["style"]),
                image_format="PNG",
                line_pad=int(rendering["line_pad"]),
                image_pad=int(rendering["image_pad"]),
            )
            expected_bytes = highlight(wrapped, lexer, formatter)
            with Image.open(io.BytesIO(expected_bytes)) as raw_expected, Image.open(row.image_path) as raw_actual:
                expected = raw_expected.convert("RGB")
                actual = raw_actual.convert("RGB")
            if actual.size != expected.size or actual.size != (int(row.image_width), int(row.image_height)):
                raise RuntimeError(f"dimension mismatch actual={actual.size} expected={expected.size}")
            if int(row.num_lines) != len(wrapped.splitlines()):
                raise RuntimeError("wrapped line count mismatch")
            if ImageChops.difference(actual, expected).getbbox() is not None:
                raise RuntimeError("pixel mismatch against source re-render")
            if actual.getbbox() is None:
                raise RuntimeError("blank image")
            checked += 1
        except Exception as exc:
            failures.append({"rq0_id": str(row.rq0_id), "error": f"{type(exc).__name__}: {exc}"})

    pd.DataFrame(failures, columns=["rq0_id", "error"]).to_csv(output.with_suffix(".failures.csv"), index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures and checked == len(metadata) else "failed",
        "rows_expected": int(len(metadata)),
        "rows_checked": checked,
        "failures": len(failures),
        "method": "full pixel equality against fresh per-snippet rendering from raw source",
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

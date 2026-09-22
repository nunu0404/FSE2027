#!/usr/bin/env python3
"""Render RQ0 code snippets to PNG images and save image metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rq0_render")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def wrap_code(code: str, wrap_column: int | None) -> str:
    code = str(code).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not wrap_column or wrap_column <= 0:
        return code
    wrapped = []
    for line in code.split("\n"):
        if len(line) <= wrap_column:
            wrapped.append(line)
            continue
        match = re.match(r"[ \t]*", line)
        indent = match.group(0) if match else ""
        remaining = line[len(indent):]
        width = max(8, wrap_column - len(indent.expandtabs(4)))
        while remaining:
            wrapped.append(indent + remaining[:width])
            remaining = remaining[width:]
    return "\n".join(wrapped)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/rq0_viability/configs/rq0_main.yaml")
    parser.add_argument("--dataset", default="experiments/rq0_viability/data/processed/pooled_313_processed.csv")
    parser.add_argument("--rendering", choices=["default", "high_legibility"], default="default")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = yaml.safe_load((repo_root / args.config).read_text(encoding="utf-8"))
    root = repo_root / config["experiment"]["root"]
    rendering = config["rendering"][args.rendering]
    logger = setup_logger(root / f"outputs/logs/render_{args.rendering}.log")
    df = pd.read_csv(repo_root / args.dataset)

    out_dir = root / "data/rendered" / args.rendering
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    render_config_hash = hashlib.sha256(json.dumps(rendering, sort_keys=True).encode("utf-8")).hexdigest()
    font_name = str((repo_root / rendering["font_name"]).resolve())
    if not Path(font_name).exists():
        font_name = str(rendering["font_name"])
    for idx, row in df.iterrows():
        rq0_id = row["rq0_id"]
        image_path = out_dir / f"{rq0_id}.png"
        render_success = True
        error_message = ""
        width = height = 0
        wrapped = wrap_code(row["raw_code"], int(rendering["wrap_column"]))
        try:
            if args.overwrite or not image_path.exists():
                try:
                    lexer = get_lexer_by_name(str(row.get("language", "java") or "java"))
                except ClassNotFound:
                    lexer = TextLexer()
                # Pygments ImageFormatter is stateful enough that reusing one
                # instance across many snippets can corrupt later PNG outputs.
                formatter = ImageFormatter(
                    font_name=font_name,
                    font_size=int(rendering["font_size"]),
                    line_numbers=bool(rendering["line_numbers"]),
                    style=str(rendering["style"]),
                    image_format="PNG",
                    line_pad=int(rendering["line_pad"]),
                    image_pad=int(rendering["image_pad"]),
                )
                png_bytes = highlight(wrapped, lexer, formatter)
                image_path.write_bytes(png_bytes)
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as exc:
            render_success = False
            error_message = f"{type(exc).__name__}: {exc}"
            logger.exception("render failed rq0_id=%s", rq0_id)
        metadata_rows.append(
            {
                "rq0_id": rq0_id,
                "dataset_name": row["dataset_name"],
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "num_lines": len(wrapped.split("\n")),
                "rendering_name": args.rendering,
                "rendering_config_hash": render_config_hash,
                "render_success": render_success,
                "error_message": error_message,
            }
        )
        if (idx + 1) % 50 == 0:
            logger.info("rendered %d/%d", idx + 1, len(df))

    metadata = pd.DataFrame(metadata_rows)
    metadata_dir = root / "outputs/render_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{args.rendering}_render_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rendering": args.rendering,
        "rendering_config": rendering,
        "rendering_config_hash": render_config_hash,
        "rows": int(len(metadata)),
        "success": int(metadata["render_success"].sum()),
        "outputs": {"image_dir": str(out_dir), "metadata": str(metadata_path)},
    }
    write_json(metadata_dir / f"{args.rendering}_render_manifest.json", manifest)
    logger.info("wrote metadata=%s success=%d/%d", metadata_path, manifest["success"], len(metadata))
    return 0 if metadata["render_success"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())

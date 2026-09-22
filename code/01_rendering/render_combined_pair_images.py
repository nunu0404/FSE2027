#!/usr/bin/env python3
"""Render labeled combined pair images for RQ0 strict-swap input audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/ANON/experiment_root/ase-2026-v1/rendering/fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def paste_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw, image: Image.Image, label: str, x: int, y: int, panel_width: int, label_height: int, font) -> int:
    bar_color = (35, 35, 35)
    text_color = (255, 255, 255)
    border_color = (120, 120, 120)
    draw.rectangle([x, y, x + panel_width, y + label_height], fill=bar_color)
    draw.text((x + 16, y + 8), label, fill=text_color, font=font)
    image_y = y + label_height
    draw.rectangle([x, image_y, x + panel_width, image_y + image.height], outline=border_color, width=2)
    canvas.paste(image, (x, image_y))
    return image_y + image.height


def make_combined(img_a: Image.Image, img_b: Image.Image, orientation: str, label_font_size: int, padding: int) -> Image.Image:
    font = load_font(label_font_size)
    label_height = label_font_size + 18
    if orientation == "horizontal":
        panel_width_a = img_a.width
        panel_width_b = img_b.width
        width = padding * 3 + panel_width_a + panel_width_b
        height = padding * 2 + label_height + max(img_a.height, img_b.height)
        canvas = Image.new("RGB", (width, height), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        paste_panel(canvas, draw, img_a, "Code A", padding, padding, panel_width_a, label_height, font)
        paste_panel(canvas, draw, img_b, "Code B", padding * 2 + panel_width_a, padding, panel_width_b, label_height, font)
        return canvas

    panel_width = max(img_a.width, img_b.width)
    width = padding * 2 + panel_width
    height = padding * 3 + label_height * 2 + img_a.height + img_b.height
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    y = padding
    y = paste_panel(canvas, draw, img_a, "Code A", padding, y, panel_width, label_height, font)
    y += padding
    paste_panel(canvas, draw, img_b, "Code B", padding, y, panel_width, label_height, font)
    return canvas


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="experiments/rq0_viability/data/pairs/compact_pooled_seed42_fold0_pairs.csv")
    parser.add_argument("--render-metadata", default="experiments/rq0_viability/outputs/render_metadata/default_render_metadata.csv")
    parser.add_argument("--output-dir", default="experiments/rq0_viability/data/rendered/combined_labeled/compact_seed42_fold0")
    parser.add_argument("--metadata-output", default="experiments/rq0_viability/outputs/render_metadata/combined_labeled_compact_metadata.csv")
    parser.add_argument("--max-horizontal-width", type=int, default=1800)
    parser.add_argument("--label-font-size", type=int, default=34)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    pairs = pd.read_csv(repo_root / args.pairs)
    render_meta = pd.read_csv(repo_root / args.render_metadata).set_index("rq0_id")
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "max_horizontal_width": args.max_horizontal_width,
        "label_font_size": args.label_font_size,
        "padding": args.padding,
        "layout_rule": "horizontal if total width <= max_horizontal_width else vertical",
        "label_text": ["Code A", "Code B"],
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    rows = []
    for idx, pair in pairs.iterrows():
        for order, a_id, b_id in [
            ("AB", pair["snippet_i"], pair["snippet_j"]),
            ("BA", pair["snippet_j"], pair["snippet_i"]),
        ]:
            img_a_path = Path(render_meta.loc[a_id, "image_path"])
            img_b_path = Path(render_meta.loc[b_id, "image_path"])
            with Image.open(img_a_path) as img_a_raw, Image.open(img_b_path) as img_b_raw:
                img_a = img_a_raw.convert("RGB")
                img_b = img_b_raw.convert("RGB")
                horizontal_width = args.padding * 3 + img_a.width + img_b.width
                orientation = "horizontal" if horizontal_width <= args.max_horizontal_width else "vertical"
                combined = make_combined(img_a, img_b, orientation, args.label_font_size, args.padding)
            filename = f"{pair['pair_id']}__{order}.png".replace("/", "_")
            out_path = out_dir / filename
            combined.save(out_path)
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "order": order,
                    "snippet_a": a_id,
                    "snippet_b": b_id,
                    "snippet_i": pair["snippet_i"],
                    "snippet_j": pair["snippet_j"],
                    "image_path": str(out_path),
                    "width": combined.width,
                    "height": combined.height,
                    "orientation": orientation,
                    "config_hash": config_hash,
                }
            )
        if (idx + 1) % 100 == 0:
            print(f"rendered pairs {idx + 1}/{len(pairs)}", flush=True)

    metadata = pd.DataFrame(rows)
    metadata_path = repo_root / args.metadata_output
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    write_json(
        metadata_path.with_suffix(".manifest.json"),
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pairs": str((repo_root / args.pairs).resolve()),
            "base_render_metadata": str((repo_root / args.render_metadata).resolve()),
            "output_dir": str(out_dir),
            "metadata": str(metadata_path),
            "rows": int(len(metadata)),
            "pairs_rendered": int(len(pairs)),
            "config": config,
            "config_hash": config_hash,
        },
    )
    print(f"wrote {metadata_path} rows={len(metadata)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

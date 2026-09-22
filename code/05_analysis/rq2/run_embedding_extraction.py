#!/usr/bin/env python3
"""Extract architecture-aware visual representations for experiment D."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
MODELS = ["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B"]
REVISIONS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "OpenGVLab/InternVL3-8B": "853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan() -> pd.DataFrame:
    grid = pd.read_csv(OUT / "rendered/metadata/grid_render_metadata.csv")
    pert = pd.read_csv(OUT / "rendered/metadata/perturbation_render_metadata.csv")
    pert = pert[~pert.condition.eq("baseline")].copy()
    grid["experiment"] = "grid"
    pert["experiment"] = "perturbation"
    columns = ["experiment", "condition", "rq0_id", "language", "image_path", "image_sha256"]
    plan = pd.concat([grid[columns], pert[columns]], ignore_index=True)
    plan["representation_condition"] = plan.experiment + "__" + plan.condition
    plan = plan.sort_values(["representation_condition", "language", "rq0_id"]).reset_index(drop=True)
    expected = 552 * 17
    if len(plan) != expected or plan.groupby("representation_condition").size().nunique() != 1:
        raise ValueError(f"embedding plan drift: {len(plan)} != {expected}")
    missing = [value for value in plan.image_path if not (ROOT / value).is_file()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]}")
    return plan


def open_rgb(path: str) -> Image.Image:
    with Image.open(ROOT / path) as image:
        return image.convert("RGB").copy()


def load_model(model_name: str):
    import torch
    if model_name.startswith("Qwen"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
        except TypeError:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
        return model.eval(), processor
    import torch.nn as nn
    from transformers import AutoImageProcessor, AutoModel
    if not hasattr(nn.Module, "all_tied_weights_keys"):
        nn.Module.all_tied_weights_keys = {}
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
        )
    return model.eval().cuda(), processor


def extract_qwen(model, processor, image: Image.Image) -> tuple[np.ndarray, np.ndarray, None, dict[str, int]]:
    import torch
    # The multimodal wrapper requires text in transformers 5.x; its image processor is the
    # exact visual preprocessing path used internally by the executed judge.
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.model.visual.dtype)
    grid = inputs["image_grid_thw"].to(device)
    with torch.no_grad():
        output = model.model.visual(pixel_values, grid_thw=grid)
    native = output.last_hidden_state.float().mean(dim=0).cpu().numpy()
    projected = output.pooler_output.float().mean(dim=0).cpu().numpy()
    return native, projected, None, {
        "native_tokens": int(output.last_hidden_state.shape[0]),
        "projected_tokens": int(output.pooler_output.shape[0]),
    }


def extract_internvl(model, processor, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    import torch
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device, dtype=model.dtype)
    with torch.no_grad():
        output = model.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state
        cls = hidden[:, 0, :]
        patches = hidden[:, 1:, :]
        side = int(math.sqrt(patches.shape[1]))
        if side * side != patches.shape[1]:
            raise ValueError(f"InternVL patch count is not square: {patches.shape}")
        projected = patches.reshape(patches.shape[0], side, side, -1)
        projected = model.pixel_shuffle(projected, scale_factor=model.downsample_ratio)
        projected = projected.reshape(projected.shape[0], -1, projected.shape[-1])
        projected = model.mlp1(projected)
    return (
        patches.float().mean(dim=1).squeeze(0).cpu().numpy(),
        projected.float().mean(dim=1).squeeze(0).cpu().numpy(),
        cls.float().squeeze(0).cpu().numpy(),
        {"native_tokens": int(patches.shape[1]), "projected_tokens": int(projected.shape[1])},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--output-tag", default="full_20260722")
    parser.add_argument("--limit", type=int, default=0, help="Deterministic plan prefix; zero means all 9,384 images.")
    args = parser.parse_args()
    plan = build_plan()
    if args.limit < 0 or args.limit > len(plan):
        raise ValueError("invalid --limit")
    if args.limit:
        # Span the complete plan instead of testing only the first rendering condition.
        positions = np.linspace(0, len(plan) - 1, args.limit, dtype=int)
        plan = plan.iloc[positions].reset_index(drop=True)

    revision = REVISIONS[args.model]
    safe = args.model.replace("/", "__")
    run_dir = OUT / "embedding/raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe}__vision__{args.output_tag}"
    npz_path = run_dir / f"{stem}.npz"
    metadata_path = run_dir / f"{stem}.csv"
    manifest_path = run_dir / f"{stem}.manifest.json"
    for path in (npz_path, metadata_path):
        if path.exists():
            raise FileExistsError(path)
    manifest = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "model_revision": revision, "dtype": "bfloat16",
        "processor_path": "same AutoProcessor/AutoImageProcessor path as executed VLM inference",
        "planned_images": len(plan), "complete_plan_images": 9384,
        "conditions": int(plan.representation_condition.nunique()),
        "primary_representation": "mean of final projected visual tokens",
        "secondary_representation": "mean of native final-layer patch tokens",
        "cls_representation": "InternVL native CLS; unavailable for Qwen architecture",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    model, processor = load_model(args.model)
    extractor = extract_qwen if args.model.startswith("Qwen") else extract_internvl
    native_values, projected_values, cls_values, records = [], [], [], []
    for index, row in enumerate(plan.itertuples(index=False), 1):
        native, projected, cls, token_counts = extractor(model, processor, open_rgb(row.image_path))
        for name, value in (("native", native), ("projected", projected)):
            if not np.isfinite(value).all() or float(np.linalg.norm(value)) == 0:
                raise RuntimeError(f"invalid {name} embedding at {row.image_path}")
        if cls is not None and (not np.isfinite(cls).all() or float(np.linalg.norm(cls)) == 0):
            raise RuntimeError(f"invalid CLS embedding at {row.image_path}")
        native_values.append(native.astype(np.float16))
        projected_values.append(projected.astype(np.float16))
        if cls is not None:
            cls_values.append(cls.astype(np.float16))
        records.append({**row._asdict(), **token_counts, "row_index": index - 1})
        if index % 25 == 0 or index == len(plan):
            print(f"embedding {args.model} images={index}/{len(plan)}", flush=True)

    arrays = {
        "native_patch_mean": np.stack(native_values),
        "projected_visual_mean": np.stack(projected_values),
    }
    if cls_values:
        arrays["native_cls"] = np.stack(cls_values)
    np.savez_compressed(npz_path, **arrays)
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    manifest.update({
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_images": len(records),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "npz_sha256": sha256(npz_path), "metadata_sha256": sha256(metadata_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

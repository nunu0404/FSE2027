#!/usr/bin/env python3
"""Run only the frozen grid baseline through the unchanged inference runner."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path("/ANON/experiment_root")
SOURCE = ROOT / "results/grounded_protocol_3lang_20260721"
RUNNER = SOURCE / "code/run_grounded_vlm.py"
BASELINE = "monokai_dark__fs20__wrap80__lnon"


def ensure_link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"wrong symlink target: {link}")
        return
    if link.exists():
        raise FileExistsError(link)
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    for name in ("config", "data", "audit", "rendered"):
        ensure_link(SOURCE / name, run_root / name)

    spec = importlib.util.spec_from_file_location("unchanged_grid_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = run_root
    original_build_plan = module.build_plan

    def baseline_plan(experiment: str):
        pairs, conditions, lookup = original_build_plan(experiment)
        if experiment != "grid" or BASELINE not in conditions:
            raise RuntimeError("frozen grid baseline is unavailable")
        return pairs, [BASELINE], lookup

    module.build_plan = baseline_plan
    argv = [
        str(RUNNER),
        "--experiment",
        "grid",
        "--model",
        args.model,
        "--output-tag",
        args.output_tag,
    ]
    if args.resume:
        argv.append("--resume")
    if args.dry_run:
        argv.append("--dry-run")
    previous = sys.argv
    try:
        sys.argv = argv
        return int(module.main())
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())

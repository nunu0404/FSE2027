#!/usr/bin/env python3
"""Run the unchanged battery runner against an isolated output tree."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path("/ANON/experiment_root")
SOURCE = ROOT / "results/rq1_model_battery_3lang_20260723"
RUNNER = SOURCE / "code/run_rq1_vlm.py"


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
    parser.add_argument("--phase", choices=("adapter", "ga0", "full"), default="full")
    parser.add_argument("--limit-calls", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    for name in ("APPROVAL_RECORD.json", "config", "data", "audit"):
        ensure_link(SOURCE / name, run_root / name)
    # The frozen protocol stores its prompt relative to the original run root
    # as ../grounded_protocol_3lang_20260721/config/prompt_B_template.txt.
    ensure_link(
        ROOT / "results/grounded_protocol_3lang_20260721",
        run_root.parent / "grounded_protocol_3lang_20260721",
    )

    spec = importlib.util.spec_from_file_location("unchanged_rq1_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = run_root

    argv = [str(RUNNER), "--model", args.model, "--phase", args.phase]
    if args.limit_calls:
        argv.extend(["--limit-calls", str(args.limit_calls)])
    if args.resume:
        argv.append("--resume")
    previous = sys.argv
    try:
        sys.argv = argv
        return int(module.main())
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())

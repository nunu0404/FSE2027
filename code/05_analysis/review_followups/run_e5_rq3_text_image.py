#!/usr/bin/env python3
"""Run the frozen RQ3 text+image protocol into an isolated E5 run tree."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path("/ANON/experiment_root")
SOURCE = ROOT / "results/grounded_protocol_3lang_20260721"
RUNNER = SOURCE / "code/run_rq3_vlm.py"
RUN_ROOT = (
    ROOT / "results/fse2027_review_defense_e1_e8_20260730/"
    "runs/e5_rq3_text_plus_image"
)


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
    rq3 = RUN_ROOT / "rq3"
    for name in ("data", "audit", "review", "rendered"):
        ensure_link(SOURCE / "rq3" / name, rq3 / name)

    spec = importlib.util.spec_from_file_location("e5_rq3_frozen_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RQ3 = rq3

    original_load_pilot = module.load_pilot

    def load_pilot_with_output_adapter():
        pilot = original_load_pilot()
        # Output-format adapter only: accepts observed unambiguous InternVL
        # misspellings while still requiring an explicit A/B verdict.
        # Prompt, generated tokens, verdict token, and logits remain unchanged.
        pilot.VERDICT_RE = re.compile(
            r"FINAL[_\s-]*VERD(?:ICT|DICT|DCT|DIC|D)\s*[:=]\s*(A|B)\b",
            re.IGNORECASE,
        )
        return pilot

    module.load_pilot = load_pilot_with_output_adapter
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())

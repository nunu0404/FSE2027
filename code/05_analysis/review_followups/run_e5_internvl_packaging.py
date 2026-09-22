#!/usr/bin/env python3
"""Run the frozen packaging protocol with an output-format-only adapter."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path("/ANON/experiment_root")
RUNNER = ROOT / "experiments/rq0_viability/scripts/run_judge_pairs.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("e5_packaging_frozen_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def parse_choice_output_adapter(text: str) -> str | None:
        cleaned = (text or "").strip()
        match = re.search(
            r"FINAL[_\s-]*VERD(?:ICT|DICT|DIC|D)\s*[:=]\s*(A|B|TIE)\b",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        return module._original_parse_choice_local(text)

    module._original_parse_choice_local = module.parse_choice_local
    module.parse_choice_local = parse_choice_output_adapter
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())

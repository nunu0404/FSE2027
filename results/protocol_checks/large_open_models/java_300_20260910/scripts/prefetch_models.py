#!/usr/bin/env python3
"""Ultra-fast model prefetcher using hf_transfer to /ANON/hf_cache."""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = "/ANON/hf_cache"

TOKEN_PATH = Path("/ANON/scratch_rq1/hf/token")
token = TOKEN_PATH.read_text().strip() if TOKEN_PATH.exists() else None

MODELS = [
    ("Qwen/Qwen2.5-VL-32B-Instruct", "7cfb30d71a1f4f49a57592323337a4a4727301da"),
    ("google/gemma-3-27b-it", "005ad3404e59d6023443cb575daa05336842228a"),
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "68faf511d618ef198fef186659617cfd2eb8e33a"),
]

print("=== [Ultra-Fast Prefetch] Starting Downloads with hf_transfer ===", flush=True)
for repo_id, rev in MODELS:
    print(f"--> Downloading {repo_id} (revision: {rev})...", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=rev,
        token=token,
        max_workers=8,
    )
    print(f"--> [COMPLETE] {repo_id} -> {path}\n", flush=True)

print("=== [Ultra-Fast Prefetch] All 3 Model Downloads Complete! ===", flush=True)

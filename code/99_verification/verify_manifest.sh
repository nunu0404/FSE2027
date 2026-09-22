#!/usr/bin/env bash
# Verify every shipped file against MANIFEST.sha256.
set -u
cd "$(dirname "$0")/../.." || exit 1

if [ ! -f MANIFEST.sha256 ]; then
  echo "MANIFEST.sha256 not found in $(pwd)" >&2
  exit 1
fi

total=$(wc -l < MANIFEST.sha256)
echo "Verifying ${total} files against MANIFEST.sha256 ..."

if sha256sum --quiet -c MANIFEST.sha256; then
  echo "OK: ${total} files verified"
  exit 0
else
  echo "FAILED: see the lines above for mismatched or missing files" >&2
  exit 1
fi

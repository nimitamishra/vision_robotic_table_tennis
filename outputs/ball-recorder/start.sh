#!/usr/bin/env bash
set -e
RECORDER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/jingxuan/Documents/Codex/2026-10-02/zhzeze/work/vision-venv/bin/python "$RECORDER_DIR/record.py" "$@"

#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /home/jingxuan/Documents/Codex/2026-10-02/zhzeze/work/vision-venv/bin/python -u run.py "$@"

#!/usr/bin/env bash
set -e
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/jingxuan/Documents/Codex/2026-10-02/zhzeze/work/vision-venv/bin/python "$app_dir/app.py" "$@"

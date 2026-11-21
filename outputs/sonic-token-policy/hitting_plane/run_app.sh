#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python "$SCRIPT_DIR/app_server.py" "$@"

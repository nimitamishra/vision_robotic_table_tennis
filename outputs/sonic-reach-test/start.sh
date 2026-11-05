#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "$0")" && pwd)"
run_dir="$here/runs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
sim_python=/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python
view_python=/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/work/vision-venv/bin/python
"$sim_python" "$here/sim_test.py" --output "$run_dir" --duration 300 --position-feedback > "$run_dir/simulator.log" 2>&1 &
sim_pid=$!
cleanup() { kill -INT "$sim_pid" 2>/dev/null || true; wait "$sim_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
printf 'Simulation viewer: http://127.0.0.1:8090\nLogs: %s\n' "$run_dir"
"$view_python" "$here/viewer.py" "$run_dir" --port 8090

#!/usr/bin/env bash
set -euo pipefail

repo="/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl"

if ! pgrep -f "python gear_sonic/scripts/run_sim_loop.py" >/dev/null; then
  gnome-terminal --title="MuJoCo G1 Simulator" -- bash -lc \
    "cd '$repo'; source .venv_sim/bin/activate; python gear_sonic/scripts/run_sim_loop.py; exec bash"
  sleep 3
fi

if ! pgrep -f "target/release/g1_deploy_onnx_ref" >/dev/null; then
  gnome-terminal --title="SONIC Control - ACTIVE" -- \
    /home/jingxuan/Documents/Codex/2026-09-15/nih/work/launch_sonic_control.sh
fi

echo "MuJoCo and SONIC Control are running."

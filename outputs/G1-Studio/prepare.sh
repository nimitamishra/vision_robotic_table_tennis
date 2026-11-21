#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python "$app_dir/prepare_gate.py" "$1"
cd /home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT="$HOME/.local/opt/TensorRT-10.13.3"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
set +u
source scripts/setup_env.sh >/dev/null
exec ./target/release/g1_deploy_onnx_ref "$1" policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --input-type manager --output-type zmq --zmq-host 127.0.0.1 --zmq-port 17576 --zmq-out-port 17577 --guarded-upper-trial --slow-preparation-test

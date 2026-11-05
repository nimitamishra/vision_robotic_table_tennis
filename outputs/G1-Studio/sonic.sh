#!/usr/bin/env bash
set -e
cd /home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT="$HOME/.local/opt/TensorRT-10.13.3"
export CUDA_HOME=/usr/local/cuda
export PATH="$TensorRT_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source scripts/setup_env.sh >/dev/null
exec ./target/release/g1_deploy_onnx_ref "$1" policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type manager --output-type all --zmq-host localhost

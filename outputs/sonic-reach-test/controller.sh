#!/usr/bin/env bash
set -euo pipefail
cd /home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT="$HOME/.local/opt/TensorRT-10.13.3"
export CUDA_HOME=/usr/local/cuda
export PATH="$TensorRT_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
set +u
source scripts/setup_env.sh >/dev/null
# Simulation only: fixed loopback interface, private input/output ports.
extra=()
if [[ "${SONIC_SIM_HOLD_INIT_UPPER_BODY:-0}" == "1" ]]; then
  extra+=(--sim-init-hold-upper-body)
fi
if [[ "${SONIC_SIM_RIGHT_ARM_SERVO:-0}" == "1" ]]; then
  extra+=(--sim-right-arm-direct-servo)
fi
exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 --zmq-port 15576 --zmq-out-port 15577 --disable-crc-check "${extra[@]}"

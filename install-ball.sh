#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$app_dir/.ball-venv"
"$app_dir/.ball-venv/bin/python" -m pip install -r "$app_dir/ball-requirements.lock.txt"
"$app_dir/.ball-venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "A working NVIDIA CUDA driver is required"'
echo "Ball model dependencies installed: $app_dir/.ball-venv"

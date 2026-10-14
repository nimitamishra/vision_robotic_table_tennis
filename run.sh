#!/bin/sh
set -eu
app_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$app_dir/.venv/bin/python" ]; then
    python_bin="$app_dir/.venv/bin/python"
else
    echo "Run $app_dir/install.sh first" >&2
    exit 1
fi
LD_LIBRARY_PATH="$app_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
exec "$python_bin" "$app_dir/app.py" "$@"

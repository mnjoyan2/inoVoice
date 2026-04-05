#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ORTCAP="$("$ROOT/.venv/bin/python" -c "import onnxruntime,os; print(os.path.join(os.path.dirname(onnxruntime.__file__),\"capi\"))")"
export LD_LIBRARY_PATH="${ORTCAP}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
exec "$ROOT/.venv/bin/uvicorn" app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" "$@"

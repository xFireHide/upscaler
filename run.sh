#!/usr/bin/env bash
# Abre a interface gráfica do upscaler (Real-ESRGAN + GFPGAN).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

exec "$PY" "$ROOT/gui.py" "$@"

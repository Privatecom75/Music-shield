#!/usr/bin/env sh
# One-command local run: creates a venv, installs deps, starts the server.
set -eu
cd "$(dirname "$0")"

PORT="${PORT:-8420}"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv 2>/dev/null || python3 -m virtualenv .venv
fi
.venv/bin/pip install -q -r requirements.txt

echo "Music Shield -> http://127.0.0.1:${PORT}"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"

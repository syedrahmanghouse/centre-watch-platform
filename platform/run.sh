#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Run ./install.sh first (Python 3.12 required).'
  exit 1
fi
exec .venv/bin/python -m uvicorn server.app:app --host "${CW_HOST:-127.0.0.1}" --port "${CW_PORT:-8787}"

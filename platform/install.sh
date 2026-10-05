#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${CW_PYTHON:-python3.12}"
if [ ! -x .venv/bin/python ]; then "$PYTHON_BIN" -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt -r requirements-vision.txt
.venv/bin/python scripts/prepare_models.py
.venv/bin/python scripts/prepare_people_model.py
printf '\nReady. Run ./run.sh and open http://127.0.0.1:8787\n'

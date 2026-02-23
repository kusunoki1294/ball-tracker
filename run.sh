#!/bin/sh
set -e

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1 || true
python -m pip install -r requirements.txt

python track_ball.py "$@"

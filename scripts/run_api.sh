#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# 8010 avoids EC2 conflicts: nginx/legacy FastAPI claim 8000; /opt/myapp uses 8045.
PORT="${APP_PORT:-8010}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --reload --log-level debug

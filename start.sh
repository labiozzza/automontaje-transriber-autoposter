#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --reload

#!/bin/sh
set -eu

python scripts/sync_models_from_s3.py

exec uvicorn api.main:app --host 0.0.0.0 --port 8000

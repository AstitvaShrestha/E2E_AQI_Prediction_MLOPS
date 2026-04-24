#!/bin/bash
# Retrain all champion models inside Docker
# Usage: bash scripts/retrain.sh
#        bash scripts/retrain.sh Delhi

set -e
cd "$(dirname "$0")/.."

CITY=${1:-""}

if [ -n "$CITY" ]; then
    echo "Retraining $CITY..."
    docker compose run --rm \
      -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
      -e DATA_DIR=/app/data \
      -e PYTHONPATH=/app \
      fastapi python src/train/trainer.py --city "$CITY"
else
    echo "Retraining all cities..."
    docker compose run --rm \
      -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
      -e DATA_DIR=/app/data \
      -e PYTHONPATH=/app \
      fastapi python src/train/trainer.py
fi
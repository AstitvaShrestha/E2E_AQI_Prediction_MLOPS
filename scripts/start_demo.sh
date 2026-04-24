#!/bin/bash
# scripts/start_demo.sh
# One-command demo startup for AQI Prediction System
# Usage: bash scripts/start_demo.sh

set -e

echo "============================================"
echo "  AQI Prediction System — Demo Startup"
echo "============================================"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker first."
    exit 1
fi

cd "$(dirname "$0")/.."

echo ""
echo "[1/5] Starting infrastructure..."
docker compose up -d postgres mlflow prometheus grafana
sleep 10

echo "[2/5] Waiting for MLflow..."
until curl -sf http://localhost:5000/health > /dev/null 2>&1; do
    sleep 5
done
echo "      MLflow ready ✓"

echo "[3/5] Starting FastAPI..."
docker compose up -d fastapi
echo "      Waiting for models to load (~90 seconds)..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
done
echo "      FastAPI ready ✓"

echo "[4/5] Starting frontend..."
docker compose up -d frontend
sleep 10
echo "      Frontend ready ✓"

echo "[5/5] Starting Airflow..."
docker compose up -d airflow-webserver airflow-scheduler airflow-dag-processor
sleep 10
echo "      Airflow starting (takes ~60 seconds)..."

echo ""
echo "============================================"
echo "  All services started successfully!"
echo "============================================"
echo ""
echo "  Streamlit dashboard : http://localhost:8501"
echo "  FastAPI docs        : http://localhost:8000/docs"
echo "  MLflow UI           : http://localhost:5000"
echo "  Airflow UI          : http://localhost:8080"
echo "  Grafana             : http://localhost:3001  (admin/admin)"
echo "  Prometheus          : http://localhost:9090"
echo ""
echo "  To stop all:  docker compose down"
echo "  To retrain:   bash scripts/retrain.sh"
echo "============================================"
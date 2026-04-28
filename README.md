# AQI Prediction System

End-to-end MLOps system forecasting India CPCB AQI 24 hours ahead for 5 major Indian cities: Delhi, Mumbai, Kolkata, Chennai, and Bengaluru.

**Github Link**:- https://github.com/AstitvaShrestha/E2E_AQI_Prediction_MLOPS/   

**DAGSHub Link (for dvc and ml artifacts)**:- https://dagshub.com/astitvashrestha1/E2E_AQI_Prediction_MLOPS/  

**Video**:- Included in repo ***Screencast MLOPs E2E AQI Prediction***.

**/docs**:- This folder included in repo contains HLD, LLD, Architecture, test plan & test cases and user manual

**Report**:- Included in repo ***Report_E2E_AQI_Prediction_DA25S013.pdf***. Covers the complete MLOps lifecycle: HLD/LLD, system architecture, DVC pipeline, Airflow DAG flows, inference pipeline design, drift detection algorithm, performance benchmarks (inference latency, API throughput, data pipeline speed), Streamlit UI screenshots, test plan with acceptance criteria, user manual, and known limitations. Average Prophet MAE: 18.32 AQI points across 5 cities.


### Stack: Prophet + SARIMA · FastAPI · Streamlit · Airflow 3.1.8 · MLflow 3.11.1 · Prometheus + Grafana · Docker Compose · DVC + DagsHub

---

## Quick Start

```bash
git clone https://github.com/AstitvaShrestha/E2E_AQI_Prediction_MLOPS
cd E2E_AQI_Prediction_MLOPS

# Set up DagsHub credentials (required even for public repos)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
# Get token at: https://dagshub.com/user/settings/tokens

# Pull data and models from DagsHub
dvc pull

# Copy environment template and fill in your API keys
cp .env.example .env

# Start entire stack
bash scripts/start_demo.sh
```

Open http://localhost:8501 for the dashboard.

---

## Prerequisites

- Docker Engine 24+ and Docker Compose v2
- Git + DVC (`pip install dvc`)
- API keys (see `.env` setup below)

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key variables:

```bash
# API Keys (required)
OPENAQ_API_KEY=your_openaq_api_key
AQICN_TOKEN=your_aqicn_token
DAGSHUB_TOKEN=your_dagshub_token

# Mailtrap SMTP (for email alerts)
MAILTRAP_USERNAME=your_mailtrap_username
MAILTRAP_PASSWORD=your_mailtrap_password
AIRFLOW_ALERT_EMAIL=your_email@example.com

# Airflow credentials
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_BASE_URL=http://airflow-webserver:8080
AIRFLOW_UID=1000  # run: id -u to get your UID

# Resource tuning (adjust based on your machine)
AIRFLOW_SCHEDULER_MEM=4g
AIRFLOW_SCHEDULER_CPUS=4.0
RETRAIN_POOL_SLOTS=1
AIRFLOW_PARALLELISM=4
AIRFLOW_MAX_ACTIVE_TASKS_PER_DAG=4
```

Get your free API keys:
- OpenAQ: https://explore.openaq.org/register
- AQICN: https://aqicn.org/api/
- DagsHub token: https://dagshub.com/user/settings/tokens

Kaggle Dataset: https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025?resource=download

---

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | None |
| FastAPI docs | http://localhost:8000/docs | None |
| MLflow UI | http://localhost:5000 | None |
| Airflow UI | http://localhost:8080 | admin / &lt;check AIRFLOW_PASSWORD in .env&gt; |
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | None |
| AlertManager | http://localhost:9093 | None |

---

## Project Structure

```
aqi_prediction/
├── src/
│   ├── config.py                  # Cities, AQI categories, pm25_to_aqi
│   ├── ingest/
│   │   ├── openaq.py              # OpenAQ v3 PM2.5 ingestion
│   │   ├── weather.py             # OpenMeteo weather ingestion
│   │   └── aqicn.py               # AQICN live AQI (inference)
│   ├── features/
│   │   ├── pipeline.py            # Feature engineering (Spark/Pandas)
│   │   └── drift.py               # KS-test drift detection
│   ├── train/
│   │   └── trainer.py             # Prophet + SARIMA training
│   └── api/
│       └── main.py                # FastAPI inference server
├── dags/
│   ├── aqi_hourly_ingest.py       # Hourly ingest + drift check DAG
│   ├── aqi_retrain.py             # Triggered retraining DAG
│   └── aqi_backfill.py            # Historical backfill DAG
├── frontend/
│   └── app.py                     # Streamlit dashboard
├── scripts/
│   ├── seed_from_kaggle.py        # Seed from Kaggle dataset
│   ├── evaluate.py                # Model evaluation report
│   ├── find_sensor_ids.py         # OPENAQ station ID discovery
│   ├── start_demo.sh              # One-command startup
│   └── retrain.sh                 # Manual retraining
├── monitoring/
│   ├── prometheus.yml             # Prometheus scrape config
│   ├── alerts.yml                 # Alert rules
│   ├── alertmanager.yml           # AlertManager routing
│   └── grafana/
│       └── dashboards/
│           └── aqi_prediction.json       # Grafana dashboard
├── docker/
│   ├── Dockerfile.api             # FastAPI + Prophet
│   ├── Dockerfile.mlflow          # MLflow server
│   ├── Dockerfile.airflow         # Airflow + dependencies
│   ├── Dockerfile.frontend        # Streamlit
│   ├── requirements.api.txt
│   ├── requirements.airflow.txt
│   └── requirements.frontend.txt
├── tests/
│   └── test_aqi_prediction.py     # 72 (unit + integration tests)
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   ├── architecture_diagram.md
│   ├── test_plan.md
│   └── user_manual.md
├── images/
│   ├── air-quality.png            # Sidebar logo
│   └── architecture_diagram.png   # System architecture image
├── reports/                       # Generated by evaluate.py
│   ├── evaluation_latest.json
│   └── evaluation_latest.csv
├── data/                          # DVC tracked — not in Git
│   ├── raw/                       # OpenAQ + Kaggle parquet files
│   ├── processed/                 # Feature-engineered parquet
│   └── baseline/                  # KS-test baseline distributions
├── mlflow/                        # DVC tracked — not in Git
│   ├── mlflow.db                  # SQLite experiment metadata
│   └── artifacts/                 # Prophet model files
├── docker-compose.yaml
├── dvc.yaml                       # DVC pipeline stages
├── dvc.lock                       # Locked pipeline state
├── pytest.ini
├── AIDisclosure.md
└── .env                           # Never committed to Git
```

---

## Running Individual Files via Docker

### Ingest data

```bash
# Ingest OpenAQ (last 2 hours, all cities)
sudo docker compose run --rm \
  -e OPENAQ_API_KEY=${OPENAQ_API_KEY} \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  fastapi python src/ingest/openaq.py

# Ingest weather (all cities)
sudo docker compose run --rm \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  fastapi python src/ingest/weather.py

# Check live AQICN readings (no save — read only)
sudo docker compose run --rm \
  -e AQICN_TOKEN=${AQICN_TOKEN} \
  -e PYTHONPATH=/app \
  fastapi python src/ingest/aqicn.py

# Seed from Kaggle CSV (one-time historical seed)
sudo docker compose run --rm \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  fastapi python scripts/seed_from_kaggle.py
```

### Build features

```bash
# Build features for all cities
sudo docker compose run --rm \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  fastapi python src/features/pipeline.py

# Check drift for all cities
sudo docker compose run --rm \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  fastapi python src/features/drift.py
```

### Train models

> **Note:** Run these with the full stack stopped (`docker compose down`) to avoid SQLite
> write conflicts on `mlflow.db`. `docker compose run` starts MLflow automatically.
> To retrain while the stack is running, use the Airflow `aqi_retrain` DAG instead.

```bash
# Train all 5 cities (Prophet only)
sudo docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  -e SKIP_SARIMA=true \
  fastapi python src/train/trainer.py

# Train single city
sudo docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  -e SKIP_SARIMA=true \
  fastapi python src/train/trainer.py --city Delhi
```

### Evaluate models

```bash
# Run evaluation on last 30 days (FastAPI must be running)
sudo docker compose exec fastapi python /app/scripts/evaluate.py

# Results saved to reports/evaluation_latest.json and .csv
cat reports/evaluation_latest.json | python -m json.tool
```

### Run tests

```bash
# Unit tests (no server needed, ~23 seconds)
pytest tests/test_aqi_prediction.py -v -m "not integration"

# Integration tests (requires docker compose up -d fastapi)
pytest tests/test_aqi_prediction.py -v -m "integration"

# All tests
pytest tests/test_aqi_prediction.py -v
```

### Reproduce DVC pipeline

```bash
# Reproduce full pipeline from scratch
dvc repro

# Run individual stages
dvc repro seed_kaggle
dvc repro build_features
dvc repro train
dvc repro evaluate

# Check pipeline status
dvc status
dvc dag
```

---

## Full Cold Start (from scratch)

```bash
# 1. Clone and pull data
git clone https://github.com/AstitvaShrestha/E2E_AQI_Prediction_MLOPS
cd E2E_AQI_Prediction_MLOPS
dvc pull

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start infrastructure
sudo docker compose up -d postgres mlflow prometheus grafana alertmanager
sleep 15

# 4. Train models (skip if dvc pull got mlflow/artifacts/)
sudo docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  fastapi python src/train/trainer.py

# 5. Start serving
sudo docker compose up -d fastapi frontend
sleep 90  # Prophet model loading takes ~90s

# 6. Start Airflow
sudo docker compose up -d airflow-webserver airflow-scheduler airflow-dag-processor

# 7. Verify
curl http://localhost:8000/health | python -m json.tool
```

---

## Retraining

```bash
# Retrain all cities
bash scripts/retrain.sh

# Retrain single city
bash scripts/retrain.sh Delhi

# Or via Docker directly
sudo docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e DATA_DIR=/app/data \
  -e PYTHONPATH=/app \
  fastapi python src/train/trainer.py --city Mumbai
```

---

## Monitoring Alerts

AlertManager (port 9093) sends email via Mailtrap when:

| Alert | Condition |
|-------|-----------|
| FastAPIDown | API unreachable for > 1 minute |
| HighErrorRate | 5xx responses > 5% for 5 minutes |
| SlowResponseTime | P95 latency > 10 seconds for 5 minutes |
| HighRequestRate | Requests > 100/second for 2 minutes |

Airflow sends email on DAG task failure (same Mailtrap sandbox).

Test AlertManager:
```bash
curl -X POST http://localhost:9093/api/v2/alerts \
  -H "Content-Type: application/json" \
  -d '[{"labels":{"alertname":"TestAlert","severity":"critical"},"annotations":{"summary":"Test","description":"Test alert"}}]'
```

---

## Data Versioning

All large files are tracked by DVC, not Git:

```bash
# Push data and models to DagsHub remote
dvc push

# Pull data and models from DagsHub remote
dvc pull

# After retraining — update DVC lock and push
dvc commit
dvc push
git add dvc.lock
git commit -m "update model artifacts after retraining"
git push
```

DagsHub remote: https://dagshub.com/astitvashrestha1/E2E_AQI_Prediction_MLOPS

---

## Stopping Services

```bash
# Stop all services
sudo docker compose down

# Stop and remove all data volumes (full reset)
sudo docker compose down -v
```

---

## Model Performance

Evaluated on last 30 days (April 2026):

| City | MAE | RMSE | MAPE | Status |
|------|-----|------|------|--------|
| Delhi | 27.56 | 39.23 | 23.5% | ✓ PASS |
| Mumbai | 17.03 | 35.01 | 20.7% | ✓ PASS |
| Kolkata | 19.07 | 32.50 | 21.9% | ✓ PASS |
| Chennai | 19.10 | 32.61 | 26.9% | ✓ PASS |
| Bengaluru | 17.54 | 27.01 | 17.4% | ✓ PASS |
| **Average** | **20.06** | **33.27** | **22.1%** | **✓ ALL PASS** |

Acceptance criterion: MAE ≤ 50 AQI points for all cities.

# Architecture Diagram — AQI Prediction System

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                   │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐    │
│  │   Kaggle    │  │  OpenAQ v3  │  │  OpenMeteo  │  │    AQICN      │    │
│  │ 2022–2025   │  │ live PM2.5  │  │   weather   │  │ inference AQI │    │
│  │  seed data  │  │ CPCB stn    │  │  forecast   │  │  lag regressor│    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬────────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              ORCHESTRATION — Apache Airflow  (port 8080)                 │
│                                                                          │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────┐     │
│  │  aqi_hourly_ingest   │  │  aqi_retrain     │  │  aqi_backfill   │     │
│  │  every hour at :05   │  │  on drift only   │  │  manual trigger │     │
│  │  ingest→drift→branch │  │  rebuild→train   │  │  date range     │     │    
│  └──────────────────────┘  └──────────────────┘  └─────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
          │                                │
          ▼                                ▼
┌────────────────────────┐  ┌─────────────────────────────────────────────┐
│    FEATURE STORE       │  │   EXPERIMENT TRACKING — MLflow (port 5000)  │
│                        │  │                                             │
│  data/raw/{city}/      │  │  Experiments: aqi_forecast_{city}           │
│    date=*.parquet      │  │  Metrics: MAE, RMSE, MAPE, Coverage         │
│  data/processed/       │  │  Artifacts: model.pr, forecast.png          │
│    features.parquet    │  │  Registry: @champion alias per city         │
│  data/baseline/        │  │  Storage: mlflow/artifacts/ (bind mount)    │
│    *_baseline*.parquet │  │                                             │
└──────────┬─────────────┘  └──────────────────┬──────────────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING  (trainer.py)                        │
│                                                                          │
│  Prophet (primary)           SARIMA (1,1,1)(1,1,1,24) (baseline)         │
│  → weather regressors        → statistical baseline only                 │
│  → multi-seasonality         → no regressors                             │
│  → uncertainty intervals     → used for MAE comparison                   │
│                                                                          │
│  Champion promotion: new model replaces @champion only if MAE improves   │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  SERVING — FastAPI  (port 8000)                          │
│                                                                          │
│  GET /forecast/{city}  →  Prophet.predict(future_df)                     │
│  GET /drift            →  KS-test parallel 5 cities                      │
│  GET /latest/{city}    →  AQICN live + disk fallback                     │
│  GET /health           →  model status check                             │
│  GET /metrics          →  Prometheus exposition                          │
│  POST /retrain/{city}  →  background retraining task                     │
│                                                                          │
│  Loads @champion models at startup via MLflow registry                   │
│  Bind mounts: ./mlflow:/app/mlflow  ./data:/app/data                     │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │  REST only — FASTAPI_URL env var
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                FRONTEND — Streamlit  (port 8501)                         │
│                                                                          │
│  AQI Forecast │ ML Pipeline Monitor │ Data Pipeline │ About              │
│  City selector, 24h chart, confidence bands, health advisory             │
│  Drift detection table, model status, MLflow/Airflow links               │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ├────────────────────────────────────┐
           ▼                                    ▼
┌──────────────────────┐          ┌─────────────────────────────────────┐
│  Prometheus (9090)   │          │  Grafana  (port 3001)               │
│  scrapes /metrics    │─────────►│  Request rate · P95 latency         │
│  every 15 seconds    │          │  Error rate · Model status          │
│  evaluates alerts    │          │  Auto-refresh every 30s             │
│                      │          └─────────────────────────────────────┘
│  Rules:              │
│  FastAPI down > 1min │          ┌─────────────────────────────────────┐
│  Error rate > 5%     │─────────►│  AlertManager  (port 9093)          │
│  P95 latency > 10s   │          │  Routes alerts → Mailtrap SMTP      │
│  Req rate > 100 rps  │          │  sandbox.smtp.mailtrap.io:2525      │
└──────────────────────┘          │  Credentials from .env              │
                                  └─────────────────────────────────────┘
```

---

## Docker Compose Services

```
Service                          Port    Image
─────────────────────────────────────────────────────────────────────────
aqi-prediction-postgres          5432    postgres:16
aqi-prediction-mlflow            5000    custom Dockerfile.mlflow
aqi-prediction-fastapi           8000    custom Dockerfile.api
aqi-prediction-frontend          8501    custom Dockerfile.frontend
aqi-prediction-airflow-webserver 8080    apache/airflow:3.1.8
aqi-prediction-airflow-scheduler  —      apache/airflow:3.1.8
aqi-prediction-airflow-dag-processor —   apache/airflow:3.1.8
aqi-prediction-prometheus        9090    prom/prometheus:latest
aqi-prediction-alertmanager      9093    prom/alertmanager:latest
aqi-prediction-grafana           3001    grafana/grafana:latest

Shared bind mounts:
  ./mlflow  → /app/mlflow   (FastAPI + MLflow — same artifact path)
  ./data    → /app/data     (FastAPI + Airflow)
  ./src     → /app/src      (FastAPI + Airflow)
  ./dags    → /opt/airflow/dags  (Airflow)

Alert routing:
Prometheus → evaluates alert rules (alerts.yml)
AlertManager → receives fired alerts → routes to Mailtrap email
Airflow → sends email on DAG failure via SMTP (same Mailtrap sandbox)
Credentials: MAILTRAP_USERNAME, MAILTRAP_PASSWORD, AIRFLOW_ALERT_EMAIL in .env
```

---

## DVC Pipeline DAG

```
seed_kaggle
  cmd: python scripts/seed_from_kaggle.py
  outs: data/raw/
      │
      ▼
build_features
  cmd: python src/features/pipeline.py
  deps: data/raw/
  outs: data/processed/  data/baseline/
      │
      ▼
train
  cmd: docker compose run fastapi python src/train/trainer.py
  deps: data/processed/
  outs: mlflow/artifacts/
      │
      ▼
evaluate
  cmd: docker compose exec fastapi python /app/scripts/evaluate.py
  deps: mlflow/artifacts/  data/processed/
  outs: reports/

Remote: DagsHub  (https://dagshub.com/astitvashrestha1/E2E_AQI_Prediction_MLOPS)
Also tracked: mlflow/mlflow.db (standalone dvc add)
```

---

## Airflow DAG Dependencies

```
aqi_hourly_ingest  (schedule: "5 * * * *")
══════════════════════════════════════════════════════
  ingest_openaq ──┐
  ingest_weather──┼──► drift_detection ──► branch
  ingest_aqicn ───┘                           │
                              ┌───────────────┴──────────────┐
                              ▼                              ▼
                        trigger_retrain               no_retrain
                              │
                              ▼  
                POST /api/v2/dags/aqi_retrain/dagRuns
                    aqi_retrain  (schedule: None)
══════════════════════════════════════════════════════
  get_cities ──► rebuild_features ──► retrain_Delhi    ──┐
                                  ──► retrain_Mumbai   ──┤
                                  ──► retrain_Kolkata  ──┼──► reload_api
                                  ──► log_summary      ──┤
                                  ──► retrain_Chennai  ──┤
                                  ──► retrain_Bengaluru ─┘
  (parallel retraining, pool: retrain_pool slots=2)

aqi_backfill  (schedule: None — manual trigger)
══════════════════════════════════════════════════════
  validate_conf ──► backfill_openaq  ──┐
                ──► backfill_weather ──┴──► log_summary
```

---

## Inference Pipeline Detail

```
GET /forecast/Delhi

  1. Validate city name → CITIES dict lookup
  2. Check champion_models["Delhi"] loaded
  3. Build future_df (next 24 hours):
     ┌──────────────────────────────────────────────┐
     │  Weather regressors:  OpenMeteo forecast API │
     │  AQI lag regressors:  AQICN live API         │
     │                       → disk fallback        │
     │                       → recent mean fallback │
     │  Calendar regressors: computed from ds       │
     │  Bonus regressors:    crop_burning,          │
     │                       festival_period        │
     └──────────────────────────────────────────────┘
  4. model.predict(future_df)
  5. Clip yhat/yhat_lower/yhat_upper to [0, 500]
  6. Build response with IST timestamps + categories
  7. Return JSON (24 ForecastHour objects)

Typical P95 latency: ~1 second
```

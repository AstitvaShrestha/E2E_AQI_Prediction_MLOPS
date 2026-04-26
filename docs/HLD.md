# High Level Design (HLD)
## AQI Prediction System

**Version:** 1.0
**Date:** April 2026
**Project:** End-to-End MLOps — AQI Forecasting for Indian Cities

---

## 1. Problem Statement

Air quality in Indian cities fluctuates significantly across hours, days,
and seasons. Citizens need advance warning of poor air quality to protect
their health and plan outdoor activities. CPCB monitors PM2.5 at hundreds
of stations but does not provide city-level 24-hour forecasts.

**Goal:** Build a production MLOps system that forecasts India CPCB AQI
24 hours ahead for 5 major Indian cities with automated ingestion,
retraining on drift, and real-time monitoring.

### Business Metrics
| Metric | Target | Achieved |
|--------|--------|---------|
| Forecast MAE | ≤ 50 AQI points | 19.84 avg |
| Inference latency P95 | < 10 seconds | ~1 second |
| Data freshness | Hourly | ✓ Airflow at :05 |
| Model drift detection | Automated | ✓ KS-test hourly |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                   │
│  Kaggle (seed) │ OpenAQ v3 │ OpenMeteo │ AQICN                  │
└───────┬────────┴─────┬─────┴─────┬─────┴──────┬─────────────────┘
        │              │           │            │
        ▼              ▼           ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER — Apache Airflow (port 8080)               │
│  aqi_hourly_ingest  │  aqi_retrain  │  aqi_backfill             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
┌───────────────────────┐  ┌────────────────────────────────────┐
│  FEATURE STORE        │  │  EXPERIMENT TRACKING               │
│  data/raw/{city}/     │  │  MLflow (port 5000)                │
│  data/processed/      │  │  → runs, metrics, artifacts        │
│  data/baseline/       │  │  → model registry @champion        │
└───────────┬───────────┘  └──────────────┬─────────────────────┘
            │                             │
            ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  SERVING LAYER — FastAPI (port 8000)                            │
│  /forecast/{city} │ /drift │ /latest/{city} │ /metrics          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRONTEND — Streamlit (port 8501)                               │
│  AQI Forecast │ ML Pipeline │ Data Pipeline │ About             │
└─────────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
┌───────────────────────┐  ┌────────────────────────────────────┐
│  METRICS COLLECTION   │  │  VISUALIZATION                     │
│  Prometheus (9090)    │  │  Grafana (port 3001)               │
│  AlertManager (9093)  │  │  → request rate, latency, errors   │
│  → scrapes /metrics   │  │  → 9 dashboard panels              │
│  → routes alerts      │  │                                    │
└───────────────────────┘  └────────────────────────────────────┘
```

---

## 3. Design Decisions

### 3.1 Model Selection — Prophet over LSTM/XGBoost

**Decision:** Facebook Prophet as primary model with SARIMA as baseline.

**Rationale:**
- Prophet natively supports multiple seasonality (daily, weekly, yearly, quarterly)
- Prophet accepts external regressors (weather, calendar) — critical for AQI
- Prophet provides calibrated uncertainty intervals — needed for confidence bands
- SARIMA serves as statistical baseline to validate Prophet performance
- When SARIMA outperforms Prophet (e.g., Bengaluru), Prophet is still used
  in production because it supports weather regressors and uncertainty quantification

### 3.2 Event-Driven DAG Architecture

**Decision:** Three separate Airflow DAGs instead of one monolithic pipeline.

**Rationale:**
```
aqi_hourly_ingest:  scheduled every hour — lightweight, always runs
aqi_retrain:        triggered only on drift — expensive, runs rarely
aqi_backfill:       manual trigger — one-time historical fill
```
- Retraining is expensive (20 min for 5 cities) — should not run hourly
- Ingestion failures should not block model serving
- Backfill is a one-time operation not part of the regular pipeline
- Separate DAGs = independent failure domains = better resilience

### 3.3 Champion/Challenger Model Promotion

**Decision:** Promote new model to `@champion` alias only if MAE improves.

**Rationale:**
- Prevents model degradation — new model must be better to replace current
- MLflow 3.x alias system used instead of deprecated stages
- Champion alias loaded at FastAPI startup via `models:/aqi_prophet_delhi@champion`
- Old champion tagged as `retired`, new challenger tagged as `challenger`

### 3.4 Drift Detection — KS-Test

**Decision:** Kolmogorov-Smirnov two-sample test on last 7 days vs baseline.

**Rationale:**
- KS-test is non-parametric — no assumption about AQI distribution shape
- Compares full distribution shape, not just mean — catches shape changes
- p < 0.05 significance threshold — standard statistical convention
- Two modes: annual baseline (default) and seasonal baseline (same-month)
- Seasonal mode reduces false positives from predictable seasonal patterns

**Known limitation:** Large baseline (2160 rows) vs small recent window (115 rows)
creates sample size bias. Fixed by subsampling baseline to match recent window size.

### 3.5 Loose Coupling — Frontend and Backend

**Decision:** Streamlit communicates with FastAPI exclusively via REST calls.

**Rationale:**
- Rubric requirement: "independent software blocks connected only via configurable REST API"
- `FASTAPI_URL` environment variable makes URL configurable
- Frontend can be replaced with React, Vue, or any HTTP client
- Backend can be tested independently via curl or Swagger UI

### 3.6 Containerization Strategy

**Decision:** All services in Docker Compose with bind mounts for data.

**Rationale:**
- Reproducibility — `docker compose up` starts entire stack
- Bind mounts (not volumes) for data — files persist on host machine
- `./mlflow:/app/mlflow` shared between FastAPI and MLflow containers
- Prophet Stan pre-compiled at Docker build time — avoids 3-minute cold start
- `workers=1` for FastAPI — Prophet models are not thread-safe

---

## 4. Data Flow

### Training Flow
```
Kaggle CSV → seed_from_kaggle.py → data/raw/{city}/date=*.parquet
OpenAQ API → openaq.py → data/raw/{city}/date=*.parquet (hourly append)
OpenMeteo  → weather.py → data/raw/{city}/weather/date=*.parquet

pipeline.py (PySpark/Pandas):
  data/raw/ → feature engineering → data/processed/{city}/features.parquet
                                  → data/baseline/{city}_baseline*.parquet

trainer.py:
  features.parquet → Prophet.fit() → MLflow experiment log
                   → SARIMA.fit()  → MLflow experiment log
                   → MAE comparison → promote @champion if better
```

### Inference Flow
```
GET /forecast/Delhi
  └── champion_models["Delhi"].predict(future_df)
        └── future_df built from:
              ├── OpenMeteo forecast (weather regressors)
              ├── AQICN live AQI (lag regressors)
              └── Calendar computation (hour, day, month features)
  └── Return 24 hourly predictions with confidence intervals
```

### Drift Detection Flow
```
Airflow aqi_hourly_ingest (every hour):
  → ingest OpenAQ + weather
  → check_all_cities() [parallel KS-test]
  → if drift detected: trigger aqi_retrain DAG
    → rebuild features → retrain → promote if better → reload FastAPI
```

---

## 5. Technology Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| ML Model | Facebook Prophet | 1.3.0 | Multi-seasonality + regressors + uncertainty |
| Baseline | SARIMA | statsmodels 0.14 | Statistical baseline, no regressors |
| Experiment Tracking | MLflow | 3.11.1 | Model registry with alias promotion |
| Data Engineering | Apache Spark + Pandas | 3.x / 2.2 | Spark for scale, Pandas fallback |
| Orchestration | Apache Airflow | 3.1.8 | DAG scheduling + event-driven retraining |
| API Framework | FastAPI | 0.110 | Async, auto-docs, Pydantic validation |
| Frontend | Streamlit | 1.32 | Rapid dashboard, pure Python |
| Monitoring | Prometheus + Grafana | latest | Industry standard metrics stack |
| Containerization | Docker Compose | 3.9 | Reproducible multi-service deployment |
| Data Versioning | DVC + DagsHub | 3.x | Model and data versioning |
| Source Control | Git | — | Code versioning |

---

## 6. Feature Engineering Design

| Feature Group | Features | Source | Purpose |
|--------------|----------|--------|---------|
| Target | y (AQI) | OpenAQ/Kaggle | Training label |
| Lag | aqi_lag_1h, 24h, 168h | Computed | Autocorrelation signal |
| Rolling | roll_mean/std/max 6h,24h,7d | Computed | Trend smoothing |
| Weather | temperature, wind, humidity | OpenMeteo | Physical dispersion model |
| Calendar | hour_sin/cos, dow_sin/cos, month_sin/cos | Computed | Cyclical encoding |
| India-specific | crop_burning, festival_period | Calendar | Oct-Nov pollution events |
| Rush hour | is_rush_hour | Calendar | Traffic emission signal |

---

## 7. Monitoring Design

### Prometheus Metrics (scraped from FastAPI /metrics)
| Metric | Type | Alert Threshold |
|--------|------|----------------|
| http_requests_total | Counter | — |
| http_request_duration_seconds | Histogram | P95 > 10s |
| aqi_models_loaded_total | Gauge | < 5 |
| up{job="fastapi"} | Gauge | == 0 for 1 min |

### Alert Routing (AlertManager + Airflow)

| Alert | Source | Trigger | Channel |
|-------|--------|---------|---------|
| FastAPIDown | AlertManager | up==0 for 1 min | Mailtrap email |
| HighErrorRate | AlertManager | 5xx > 5% for 5 min | Mailtrap email |
| SlowResponseTime | AlertManager | P95 > 10s | Mailtrap email |
| HighRequestRate | AlertManager | > 100 req/s for 2 min | Mailtrap email |
| DAG failure | Airflow built-in | task failure | Mailtrap email |
| Drift detected | Airflow built-in | is_drifted=True | Mailtrap email |

Two alert channels used intentionally:
- AlertManager: infrastructure alerts (API health, latency, error rate)
- Airflow: pipeline alerts (DAG failures, drift, retraining outcomes)

### Grafana Dashboard Panels
1. FastAPI Status (UP/DOWN)
2. Active Champion Models (0–5)
3. Memory Usage (MB)
4. Request Rate by endpoint (req/s)
5. Response Time P95 by endpoint (seconds)
6. Forecast Endpoint Latency Gauge (green <5s, red >10s)
7. Total Requests by Endpoint (bar chart)
8. Error Rate 5xx (time series)
9. Requests by Status Code (pie chart)

---

## 8. Reproducibility

Every experiment is reproducible via:
- **Git commit hash** — exact code state
- **DVC lock file** — exact data and model artifact versions
- **MLflow run ID** — exact hyperparameters and metrics

```bash
# Reproduce any historical state
git checkout <commit_hash>
dvc pull
docker compose up
```

**DVC Pipeline Stages:**
```
seed_kaggle → build_features → train → evaluate
```

Tracked artifacts:
- `data/raw/` — OpenAQ + Kaggle raw parquet files (~131MB)
- `data/processed/` — Feature-engineered parquet files (~11MB)
- `data/baseline/` — KS-test baseline distributions (~1.7MB)
- `mlflow/artifacts/` — Prophet model files (~808MB)
- `mlflow/mlflow.db` — Experiment metadata (~1MB)

---

## 9. Security Considerations

- API keys stored in `.env` file — never committed to Git
- `.env` in `.gitignore`
- Docker containers run as non-root where possible
- CORS enabled for all origins (development) — restrict in production
- No authentication on API endpoints (internal service assumption)
- MLflow `--allowed-hosts "*"` — restrict to known hosts in production

---

## 10. Known Limitations and Production Improvements

| Limitation | Current State | Production Fix |
|------------|--------------|----------------|
| Coverage 70% | Prophet CI underestimated | Conformal prediction |
| Seasonal drift FP | Mitigated with seasonal baseline | Same-season baseline comparison |
| AQICN single station | Geo-coordinate nearest station | Multi-station median |
| Prophet not thread-safe | workers=1 | Multiple uvicorn instances with model pre-load |
| SARIMA memory 4-6GB | Subsample to 2 years | Distributed training |
| KS-test sample bias | Subsample baseline to recent N | Bootstrap KS-test |
| No ground truth feedback | N/A | Log actual AQI next day, compute real-world MAE |

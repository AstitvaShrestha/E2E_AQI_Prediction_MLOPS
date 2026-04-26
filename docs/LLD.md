# Low Level Design (LLD)
## AQI Prediction System — API Endpoint Specifications

**Base URL:** `http://localhost:8000`
**Framework:** FastAPI 0.110+
**Model:** Facebook Prophet with weather regressors
**Authentication:** None (internal service)

---

## 1. Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | System health and model status |
| GET | /cities | List supported cities |
| GET | /forecast/{city} | 24-hour AQI forecast |
| GET | /latest/{city} | Current live AQI |
| GET | /drift | KS-test drift detection |
| POST | /retrain/{city} | Trigger background retraining |
| POST | /reload-models | Reload champion models from MLflow |
| GET | /metrics | Prometheus metrics |

---

## 2. Endpoint Specifications

### GET /health

Health check — returns loaded models and system status.

**Request:** No parameters required.

**Response 200 OK:**
```json
{
  "status": "healthy",
  "models_loaded": ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"],
  "models_missing": [],
  "mlflow_uri": "http://mlflow:5000",
  "timestamp": "2026-04-25T10:00:00.000000"
}
```

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| status | string | "healthy" — all models loaded; "degraded" — some missing |
| models_loaded | list[str] | Cities with champion models in memory |
| models_missing | list[str] | Cities without loaded models |
| mlflow_uri | string | MLflow tracking server URI |
| timestamp | string | UTC ISO 8601 timestamp |

---

### GET /cities

List available cities with coordinates and model status.

**Request:** No parameters required.

**Response 200 OK:**
```json
{
  "Delhi": {
    "lat": 28.6139,
    "lon": 77.209,
    "timezone": "Asia/Kolkata",
    "model_loaded": true
  },
  "Mumbai": {
    "lat": 19.076,
    "lon": 72.8777,
    "timezone": "Asia/Kolkata",
    "model_loaded": true
  }
}
```

**Output Fields (per city):**
| Field | Type | Description |
|-------|------|-------------|
| lat | float | Latitude coordinate |
| lon | float | Longitude coordinate |
| timezone | string | City timezone (always "Asia/Kolkata") |
| model_loaded | bool | Whether champion model is in memory |

---

### GET /forecast/{city}

Generate 24-hour AQI forecast using champion Prophet model.

**Path Parameters:**
| Parameter | Type | Required | Validation |
|-----------|------|----------|------------|
| city | string | Yes | Must be one of: Delhi, Mumbai, Kolkata, Chennai, Bengaluru |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| horizon | int | 24 | Forecast hours (1–48) |

**Response 200 OK:**
```json
{
  "city": "Delhi",
  "generated_at": "2026-04-25T10:00:00.000000",
  "horizon_hours": 24,
  "current_aqi": 152,
  "forecast": [
    {
      "timestamp": "2026-04-25T10:00:00",
      "timestamp_ist": "2026-04-25T15:30:00+05:30",
      "aqi": 155,
      "aqi_lower": 120,
      "aqi_upper": 190,
      "category": {
        "label": "Moderate",
        "color": "#ffff00",
        "advice": "Breathing discomfort to people with lung/heart disease."
      }
    }
  ],
  "model": {
    "name": "aqi_prophet_delhi",
    "alias": "champion"
  }
}
```

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| city | string | City name |
| generated_at | string | UTC timestamp of forecast generation |
| horizon_hours | int | Number of hours forecast |
| current_aqi | float or null | Live AQI from AQICN (null if unavailable) |
| forecast | list[ForecastHour] | Hourly predictions |
| model.name | string | MLflow registered model name |
| model.alias | string | Model alias used ("champion") |

**ForecastHour Object:**
| Field | Type | Description |
|-------|------|-------------|
| timestamp | string | UTC ISO timestamp |
| timestamp_ist | string | IST timestamp (+05:30) |
| aqi | int | Predicted AQI (clipped 0–500) |
| aqi_lower | int | 95% confidence interval lower bound |
| aqi_upper | int | 95% confidence interval upper bound |
| category | Category | AQI category classification |

**Category Object:**
| Field | Type | Description |
|-------|------|-------------|
| label | string | Good / Satisfactory / Moderate / Poor / Very Poor / Severe |
| color | string | Hex color code for UI display |
| advice | string | Health advisory text |

**Error Responses:**
| Status | Condition | Response Body |
|--------|-----------|---------------|
| 404 | City not in supported list | `{"detail": "City 'X' not found"}` |
| 503 | Champion model not loaded | `{"detail": "No champion model loaded for X"}` |
| 500 | Forecast generation failed | `{"detail": "Forecast generation failed: <error>"}` |

**Inference Pipeline:**
```
1. Validate city name
2. Check champion model loaded
3. Build future dataframe (next 24 hours)
   a. Weather regressors: OpenMeteo forecast API
   b. AQI lag regressors: AQICN live → disk fallback → mean fallback
   c. Calendar regressors: computed from timestamps
   d. Bonus regressors: crop_burning, festival_period from calendar
4. model.predict(future_df)
5. Clip predictions to [0, 500]
6. Format response with IST timestamps and categories
```

---

### GET /latest/{city}

Get current AQI reading for a city from AQICN.

**Path Parameters:**
| Parameter | Type | Required | Validation |
|-----------|------|----------|------------|
| city | string | Yes | Must be supported city |

**Response 200 OK:**
```json
{
  "city": "Delhi",
  "aqi": 152,
  "category": {
    "label": "Moderate",
    "color": "#ffff00",
    "advice": "Breathing discomfort to people with lung/heart disease."
  },
  "timestamp": "2026-04-25T10:00:00.000000",
  "source": "aqicn_live"
}
```

**Source Values:**
| Value | Meaning |
|-------|---------|
| aqicn_live | Fresh reading from AQICN API |
| disk_cache | Last saved reading from disk parquet files |

**Fallback Chain:**
```
1. AQICN live API (primary)
2. Last known value from data/raw/{city}/date=*.parquet (fallback)
3. HTTP 503 if neither available
```

**Error Responses:**
| Status | Condition |
|--------|-----------|
| 404 | City not supported |
| 503 | No AQI data available from any source |

---

### GET /drift

Run KS-test drift detection for all 5 cities in parallel.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| seasonal | bool | false | Use same-month historical baseline to reduce seasonal false positives |

**Response 200 OK:**
```json
{
  "checked_at": "2026-04-25T10:00:00.000000",
  "cities_drifted": ["Mumbai", "Kolkata"],
  "retrain_required": true,
  "baseline_type": "annual",
  "results": {
    "Delhi": {
      "city": "Delhi",
      "p_value": 0.945129,
      "ks_stat": 0.0696,
      "is_drifted": false,
      "baseline_mean": 150.0,
      "recent_mean": 152.4,
      "recent_n": 115,
      "baseline_type": "annual",
      "checked_at": "2026-04-25T10:00:00+00:00",
      "error": null
    }
  }
}
```

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| checked_at | string | UTC timestamp of drift check |
| cities_drifted | list[str] | Cities where drift detected |
| retrain_required | bool | True if any city drifted |
| baseline_type | string | "annual" or "seasonal" |
| results | dict | Per-city KS-test results |

**Per-City Result Fields:**
| Field | Type | Description |
|-------|------|-------------|
| p_value | float | KS-test p-value (< 0.05 = drift) |
| ks_stat | float | KS statistic (0–1, higher = more drift) |
| is_drifted | bool | True if p_value < 0.05 |
| baseline_mean | float | Mean AQI in baseline distribution |
| recent_mean | float | Mean AQI in last 7 days |
| recent_n | int | Number of recent hourly samples used |
| error | string or null | Error message if check failed |

**Algorithm:**
```
For each city (parallel, ThreadPoolExecutor):
  1. Load baseline: data/baseline/{city}_baseline[_mMM].parquet
  2. Load recent: data/raw/{city}/date=*.parquet (last 7 days)
     - Aggregate to hourly mean (fix for multi-station data)
  3. Subsample baseline to len(recent) to remove sample-size bias
  4. scipy.stats.ks_2samp(baseline_sample, recent)
  5. is_drifted = p_value < 0.05
```

**Error Responses:**
| Status | Condition |
|--------|-----------|
| 500 | Drift detection failed |

---

### POST /retrain/{city}

Trigger background model retraining. Returns immediately.

**Path Parameters:**
| Parameter | Type | Required |
|-----------|------|----------|
| city | string | Yes |

**Response 200 OK:**
```json
{
  "status": "retraining_started",
  "city": "Delhi",
  "message": "Retraining triggered for Delhi in background"
}
```

**Background Task:**
```
1. Load features from data/processed/{city}/features.parquet
2. Run train_prophet(city, df) — logs to MLflow
3. Run evaluate_and_promote(city, mae, version)
4. If promoted: reload champion model into memory
```

---

### POST /reload-models

Reload all champion models from MLflow registry.

**Request:** No parameters required.

**Response 200 OK:**
```json
{
  "status": "reloaded",
  "models_loaded": ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"],
  "timestamp": "2026-04-25T10:00:00.000000"
}
```

---

### GET /metrics

Prometheus metrics endpoint — scraped every 15 seconds by Prometheus.

**Response 200 OK:** Prometheus text format

**Key Metrics Exposed:**
| Metric | Type | Description |
|--------|------|-------------|
| http_requests_total | Counter | Requests by handler, method, status |
| http_request_duration_seconds | Histogram | Latency by handler |
| http_request_size_bytes | Summary | Request size by handler |
| http_response_size_bytes | Summary | Response size by handler |
| aqi_models_loaded_total | Gauge | Number of champion models in memory |

---

## 3. AQI Category Mapping (India CPCB)

| Range | Label | Color | Advice |
|-------|-------|-------|--------|
| 0–50 | Good | #00b050 | No health implications |
| 51–100 | Satisfactory | #92d050 | Minor discomfort for sensitive people |
| 101–200 | Moderate | #ffff00 | Breathing discomfort for lung/heart patients |
| 201–300 | Poor | #ff9900 | Breathing discomfort for most people |
| 301–400 | Very Poor | #ff0000 | Respiratory illness on prolonged exposure |
| 401–500 | Severe | #c00000 | Health impact even on light physical activity |

---

## 4. Supported Cities

| City | Latitude | Longitude | OpenAQ Locations | AQI Profile |
|------|----------|-----------|-----------------|-------------|
| Delhi | 28.6139 | 77.2090 | 11 stations | severe_winter |
| Mumbai | 19.0760 | 72.8777 | 15 stations | coastal_moderate |
| Kolkata | 22.5726 | 88.3639 | 9 stations | industrial_high |
| Chennai | 13.0827 | 80.2707 | 7 stations | coastal_low |
| Bengaluru | 12.9716 | 77.5946 | 10 stations | traffic_moderate |

---

## 5. Frontend-Backend Interface

The Streamlit frontend (port 8501) communicates with FastAPI (port 8000)
exclusively via HTTP REST calls. No shared state or database.

```
FASTAPI_URL env var (configurable)
Default: http://localhost:8000  (local development)
Docker:  http://fastapi:8000    (container network)
```

**Loose Coupling Implementation:**
- Frontend never imports backend code
- All data flows through REST endpoints
- Frontend can be replaced with any HTTP client
- Backend can be called independently via curl or API docs

---

## 6. Model Loading

Models are loaded at FastAPI startup via `lifespan` context manager:

```python
model_uri = f"models:/aqi_prophet_{city}@champion"
model = mlflow.prophet.load_model(model_uri)
```

MLflow resolves the `@champion` alias → fetches artifact path from
`mlflow/mlflow.db` → loads model from `mlflow/artifacts/`.

Both FastAPI and MLflow containers mount `./mlflow:/app/mlflow` — same
bind mount ensures artifact files are accessible from both containers.

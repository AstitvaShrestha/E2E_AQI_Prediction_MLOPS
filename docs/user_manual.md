# AQI Prediction System — User Manual

## What is this system?

The AQI Prediction System forecasts air quality for 5 major Indian cities 24 hours in advance using machine learning. It helps you plan outdoor activities based on predicted air quality.

**Supported cities:** Delhi, Mumbai, Kolkata, Chennai, Bengaluru

---

## Quick Start

1. Open your browser and go to: **http://localhost:8501**
2. The dashboard loads automatically — no login required

---

## Page 1 — AQI Forecast (Main Page)

### How to use:
1. Select a city from the **"Select City"** dropdown
2. Read the **Current AQI** number (top left card)
3. Check the **Health Advisory** for recommended actions
4. View the **24-Hour Forecast** chart for upcoming air quality

### Understanding the forecast chart:
- **Blue line** — predicted AQI for the next 24 hours
- **Shaded blue area** — uncertainty range (95% confidence interval)
- **Dotted horizontal lines** — AQI category boundaries

### Understanding the metric cards:
| Card | Meaning |
|------|---------|
| Current AQI | Live reading from monitoring stations |
| Next 6h Avg | Average predicted AQI for next 6 hours |
| 24h Peak | Highest predicted AQI in next 24 hours |
| 24h Lowest | Lowest predicted AQI in next 24 hours |

### AQI Scale (India CPCB Standard):
| Category | AQI Range | Health Impact | Recommended Action |
|----------|-----------|---------------|-------------------|
| 🟢 Good | 0–50 | No health implications | Ideal for outdoor activities |
| 🟡 Satisfactory | 51–100 | Minor discomfort for sensitive people | Generally safe |
| 🟡 Moderate | 101–200 | Breathing discomfort for lung/heart patients | Sensitive groups limit outdoor exposure |
| 🟠 Poor | 201–300 | Breathing discomfort for most people | Avoid prolonged outdoor exposure |
| 🔴 Very Poor | 301–400 | Respiratory illness on prolonged exposure | Stay indoors |
| 🔴 Severe | 401–500 | Serious health impact | Avoid all outdoor activity |

### All Cities Summary:
Scroll down to see current AQI for all 5 cities at a glance.

---

## Page 2 — ML Pipeline Monitor

### Champion Models:
Shows which cities have trained models loaded:
- **Loaded ✓** (green) — model ready, forecasts available
- **Missing ✗** (red) — model not loaded, forecasts unavailable

### Data Drift Detection:
Click **"Run Drift Detection"** to check if recent air quality patterns differ from historical patterns.

| Column | Meaning |
|--------|---------|
| p-value | Lower = stronger evidence of drift. Threshold: 0.05 |
| KS Statistic | Higher = larger distributional difference |
| Status | OK ✓ = no drift, DRIFT ⚠ = retraining recommended |
| Baseline Mean | Historical average AQI |
| Recent Mean | Last 7 days average AQI |

**Seasonal mode:** Click drift with `?seasonal=true` to compare against same-month historical data — reduces false positives from seasonal patterns.

### MLflow Links:
Click **"Open MLflow at localhost:5000"** to view experiment tracking, model metrics, and training history.

---

## Page 3 — Data Pipeline

Shows data ingestion architecture and pipeline status.

### Data Sources:
| Source | Purpose | Update Frequency |
|--------|---------|-----------------|
| Kaggle Dataset | Historical training data (2022–2025) | One-time seed |
| OpenAQ v3 API | Live PM2.5 from CPCB stations | Every hour |
| OpenMeteo | Weather forecast for regressors | Every hour |
| AQICN | Current AQI for inference lag regressor | Every forecast |

### Pipeline Management Links:
- **Airflow UI** — http://localhost:8080 (admin/admin)
- **Grafana** — http://localhost:3001 (admin/admin)
- **Prometheus** — http://localhost:9090

---

## Page 4 — About

Contains system information, AQI scale reference, technology stack, and this user manual.

---

## Refreshing Data

- Data **auto-refreshes every 5 minutes**
- Click **"🔄 Refresh Data"** in the sidebar for immediate refresh
- Forecasts update hourly as new OpenAQ data is ingested

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| "❌ API Offline" in sidebar | FastAPI not running — run `bash scripts/start_demo.sh` |
| "⚠️ 0/5 models" warning | Models not loaded — run trainer inside Docker |
| Forecast shows stale times | Click Refresh Data button |
| Drift detection spinner hangs | Normal — KS-test takes 30–60 seconds for 5 cities |
| City shows N/A for current AQI | AQICN API temporarily unavailable — disk cache used |
| Graph x-axis shows wrong timezone | All times shown in IST (UTC+5:30) |
| No alert emails received | Check AlertManager at localhost:9093, verify Mailtrap credentials in .env |

---

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | None required |
| API Docs | http://localhost:8000/docs | None required |
| MLflow UI | http://localhost:5000 | None required |
| Airflow UI | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | None required |
| AlertManager | http://localhost:9093 | None required |

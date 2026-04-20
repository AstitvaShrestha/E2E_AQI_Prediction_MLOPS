"""
Streamlit dashboard for AQI Prediction system.
Calls FastAPI endpoints for all data — loose coupling as required by rubric.

Pages:
  1. AQI Forecast     — current AQI + 24h forecast per city
  2. ML Pipeline      — drift detection + model metrics
  3. Data Pipeline    — ingestion status + data coverage
  4. About            — system architecture + user manual
"""


import os
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- Config--------

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


# India CPCB AQI category colors
CATEGORY_COLORS = {
    "Good":        "#00b050",
    "Satisfactory":"#92d050",
    "Moderate":    "#ffff00",
    "Poor":        "#ff9900",
    "Very Poor":   "#ff0000",
    "Severe":      "#c00000",
}

CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]


# ----Page Config----------

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon   = "🌬️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# --- Custom CSS--------

st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4466;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-top: 8px;
    }
    .status-ok {
        color: #00b050;
        font-weight: bold;
    }
    .status-drift {
        color: #ff4444;
        font-weight: bold;
    }
    .info-box {
        background: #1e2130;
        border-left: 4px solid #4a9eff;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ------API Helpers-------

@st.cache_data(ttl=300) #cache 5 mins
def get_forecast(city):
    """Fetch 24h forecast from FastAPI."""
    
    try:
        resp = requests.get(
            f"{FASTAPI_URL}/forecast/{city}",
            timeout = 30,
        )

        if resp.status_code == 200:
            return resp.json()
        
        st.warning(f"Forecast API returned {resp.status_code} for {city}")
        return None
    
    except requests.RequestException as e:
        st.error(f"Cannot reach FastAPI at {FASTAPI_URL}: {e}")
        return None
    

@st.cache_data(ttl=60) #cache 1 min
def get_latest_aqi(city):
    """Fetch latest AQI from FastAPI."""
    
    try:
        resp = requests.get(
            f"{FASTAPI_URL}/latest/{city}",
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()
        
        return None

    except requests.RequestException:
        return None


@st.cache_data(ttl=300) #cache 5 mins
def get_drift_status():
    """Fetch drift detection results from FastAPI."""

    try:
        resp = requests.get(
            f"{FASTAPI_URL}/drift",
            timeout=60
        )

        if resp.status_code == 200:
            return resp.json()
        
        return None
    
    except requests.RequestException as e:
        st.error(f"Drift check failed: {e}")
        return None



@st.cache_data(ttl=60)
def get_health():
    """Fetch API health status."""

    try:
        resp = requests.get(
            f"{FASTAPI_URL}/health",
            timeout = 10,
        )

        if resp.status_code == 200:
            return resp.json()

        return None

    except requests.RequestException as e:
        return None


def utc_to_ist(utc_str):
    """Convert UTC ISO string to IST for display."""

    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))

        # Add 5:30 for IST
        from datetime import timezone, timedelta

        IST = timezone(timedelta(hours=5, minutes=30))
        ist = dt.astimezone(IST)

        return ist.strftime("%d %b %H:%M IST")

    except:
        return utc_str[:16]



# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "/home/astitva/Documents/IITM/MLOps/Assignments/E2E Project/aqi_prediction/images/air-quality.png",
        width=80
    )
    st.title("AQI Prediction")
    st.caption("24-hour forecast for Indian cities")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🌬️ AQI Forecast",
         "🤖 ML Pipeline",
         "📊 Data Pipeline",
         "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    # API status indicator
    health = get_health()
    if health:
        loaded = len(health.get("models_loaded", []))
        missing = len(health.get("models_missing", []))
        if missing == 0:
            st.success(f"✅ API Online — {loaded}/5 models")
        else:
            st.warning(f"⚠️ API Online — {loaded}/5 models")
    else:
        st.error("❌ API Offline")

    st.caption(
        f"Last refresh: "
        f"{datetime.utcnow().strftime('%H:%M UTC')}"
    )

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — AQI Forecast
# ══════════════════════════════════════════════════════════════════════════

if page == "🌬️ AQI Forecast":

    st.title("🌬️ 24-Hour AQI Forecast")
    st.caption(
        "Real-time air quality predictions using Prophet time series model. "
        "All times shown in IST."
    )

    # City selector
    col1, col2 = st.columns([2, 3])
    with col1:
        selected_city = st.selectbox(
            "Select City",
            CITIES,
            index=0,
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "💡 AQI scale follows India CPCB standard. "
            "Good (0-50) → Severe (401-500)"
        )

    # Fetch data
    with st.spinner(f"Fetching forecast for {selected_city}..."):
        forecast_data = get_forecast(selected_city)
        latest_data   = get_latest_aqi(selected_city)

    if not forecast_data:
        st.error(
            f"Could not fetch forecast for {selected_city}. "
            "Check if FastAPI is running."
        )
        st.stop()

    # ── Current AQI metric ────────────────────────────────────────────────
    st.subheader("Current Air Quality")

    current_aqi = forecast_data.get("current_aqi")
    if current_aqi:
        first_forecast = forecast_data["forecast"][0]
        category       = first_forecast["category"]
        cat_label      = category.get("label", "Unknown")
        cat_color      = category.get("color", "#gray")
        cat_advice     = category.get("advice", "")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Current AQI</p>
                <p class="metric-value" style="color:{cat_color}">
                    {int(current_aqi)}
                </p>
                <span class="category-badge"
                      style="background:{cat_color}20;
                             color:{cat_color}">
                    {cat_label}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            forecast_list = forecast_data["forecast"]
            next_6h_avg   = sum(
                f["aqi"] for f in forecast_list[:6]
            ) / 6
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Next 6h Avg</p>
                <p class="metric-value" style="color:#4a9eff">
                    {int(next_6h_avg)}
                </p>
                <span class="category-badge"
                      style="background:#4a9eff20; color:#4a9eff">
                    Forecast
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            max_aqi = max(f["aqi"] for f in forecast_list)
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">24h Peak AQI</p>
                <p class="metric-value" style="color:#ff9900">
                    {max_aqi}
                </p>
                <span class="category-badge"
                      style="background:#ff990020; color:#ff9900">
                    Maximum
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            min_aqi = min(f["aqi"] for f in forecast_list)
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">24h Lowest AQI</p>
                <p class="metric-value" style="color:#00b050">
                    {min_aqi}
                </p>
                <span class="category-badge"
                      style="background:#00b05020; color:#00b050">
                    Minimum
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <strong>Health Advisory:</strong> {cat_advice}
        </div>
        """, unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────────────────────────────
    st.subheader("24-Hour Forecast")

    forecast_list = forecast_data["forecast"]
    df_forecast   = pd.DataFrame(forecast_list)
    df_forecast["time_ist"] = df_forecast["timestamp"].apply(utc_to_ist)

    fig = go.Figure()

    # Confidence interval
    fig.add_trace(go.Scatter(
        x    = df_forecast["time_ist"],
        y    = df_forecast["aqi_upper"],
        mode = "lines",
        line = dict(width=0),
        name = "Upper bound",
        showlegend = False,
    ))
    fig.add_trace(go.Scatter(
        x          = df_forecast["time_ist"],
        y          = df_forecast["aqi_lower"],
        fill       = "tonexty",
        fillcolor  = "rgba(74, 158, 255, 0.15)",
        mode       = "lines",
        line       = dict(width=0),
        name       = "95% Confidence",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x    = df_forecast["time_ist"],
        y    = df_forecast["aqi"],
        mode = "lines+markers",
        name = "Forecast AQI",
        line = dict(color="#4a9eff", width=2.5),
        marker = dict(size=6),
    ))

    # AQI category threshold lines
    thresholds = [
        (50,  "Good",         "#00b050"),
        (100, "Satisfactory", "#92d050"),
        (200, "Moderate",     "#ffff00"),
        (300, "Poor",         "#ff9900"),
        (400, "Very Poor",    "#ff0000"),
    ]
    for threshold, label, color in thresholds:
        fig.add_hline(
            y            = threshold,
            line_dash    = "dot",
            line_color   = color,
            line_width   = 1,
            annotation_text = label,
            annotation_position = "right",
        )

    fig.update_layout(
        title      = f"{selected_city} — AQI Forecast (IST)",
        xaxis_title = "Time (IST)",
        yaxis_title = "AQI (India CPCB)",
        yaxis      = dict(range=[0, 500]),
        plot_bgcolor  = "#1e2130",
        paper_bgcolor = "#0e1117",
        font          = dict(color="#c9d1d9"),
        legend        = dict(
            bgcolor     = "#1e2130",
            bordercolor = "#3d4466",
            borderwidth = 1,
        ),
        hovermode = "x unified",
        height    = 450,
    )
    st.plotly_chart(fig, width="stretch")

    # ── Hourly breakdown table ─────────────────────────────────────────────
    st.subheader("Hourly Breakdown")

    df_display = df_forecast[
        ["time_ist", "aqi", "aqi_lower", "aqi_upper"]
    ].copy()
    df_display.columns = ["Time (IST)", "AQI", "Lower", "Upper"]

    # Color rows by category
    def color_aqi(val):
        if val <= 50:   return "background-color: #00b05030"
        if val <= 100:  return "background-color: #92d05030"
        if val <= 200:  return "background-color: #ffff0030"
        if val <= 300:  return "background-color: #ff990030"
        if val <= 400:  return "background-color: #ff000030"
        return "background-color: #c0000030"

    styled = df_display.style.applymap(
        color_aqi, subset=["AQI"]
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── All cities summary ─────────────────────────────────────────────────
    st.subheader("All Cities — Current AQI")

    cols = st.columns(5)
    for i, city in enumerate(CITIES):
        with cols[i]:
            city_data = get_latest_aqi(city)
            if city_data:
                aqi      = int(city_data["aqi"])
                category = city_data["category"]
                label    = category.get("label", "")
                color    = category.get("color", "#gray")
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{city}</p>
                    <p class="metric-value" style="color:{color};
                       font-size:1.8rem">{aqi}</p>
                    <span style="color:{color};font-size:0.8rem">
                        {label}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{city}</p>
                    <p style="color:#666">N/A</p>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ML Pipeline
# ══════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Pipeline":

    st.title("🤖 ML Pipeline Monitor")
    st.caption(
        "Experiment tracking, model registry, and drift detection. "
        "Champion models are automatically promoted when MAE improves."
    )

    # ── Model registry ────────────────────────────────────────────────────
    st.subheader("Champion Models — MLflow Registry")

    health = get_health()
    if health:
        models_loaded  = health.get("models_loaded", [])
        models_missing = health.get("models_missing", [])

        cols = st.columns(5)
        for i, city in enumerate(CITIES):
            with cols[i]:
                loaded = city in models_loaded
                color  = "#00b050" if loaded else "#ff4444"
                status = "Loaded ✓" if loaded else "Missing ✗"
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{city}</p>
                    <p style="color:{color};font-size:1rem;
                       font-weight:bold">{status}</p>
                    <p style="color:#666;font-size:0.75rem">
                        @champion
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # ── Drift detection ───────────────────────────────────────────────────
    st.subheader("Data Drift Detection — KS Test")
    st.markdown("""
    <div class="info-box">
        The Kolmogorov-Smirnov test compares the full AQI distribution
        of the last 7 days against the 3-year training baseline.
        p-value &lt; 0.05 indicates statistically significant drift.
    </div>
    """, unsafe_allow_html=True)

    if "drift_data" not in st.session_state:
        st.session_state.drift_data = None

    if st.button("🔍 Run Drift Detection", type="primary"):
        st.cache_data.clear()
        with st.spinner("Running KS-test for all cities..."):
            st.session_state.drift_data = get_drift_status()
    
    # if st.button("🔍 Run Drift Detection", type="primary"):
    #     st.cache_data.clear()

    # with st.spinner("Running KS-test for all cities..."):
    #     drift_data = get_drift_status()

    drift_data = st.session_state.drift_data

    if drift_data is None:
        st.info("Click 'Run Drift Detection' to check for data drift")

    elif drift_data:
        results        = drift_data.get("results", {})
        cities_drifted = drift_data.get("cities_drifted", [])
        checked_at     = drift_data.get("checked_at", "")

        st.caption(f"Last checked: {utc_to_ist(checked_at)}")

        if cities_drifted:
            st.warning(
                f"⚠️ Drift detected in: {', '.join(cities_drifted)} "
                f"— Retraining recommended"
            )
        else:
            st.success("✅ No drift detected — all cities within baseline")

        # Drift table
        rows = []
        for city, r in results.items():
            if r.get("error"):
                rows.append({
                    "City":           city,
                    "p-value":        "ERROR",
                    "KS Statistic":   "ERROR",
                    "Status":         "Error",
                    "Baseline Mean":  "-",
                    "Recent Mean":    "-",
                    "Samples":        "-",
                })
            else:
                rows.append({
                    "City":          city,
                    "p-value":       f"{r['p_value']:.4f}",
                    "KS Statistic":  f"{r['ks_stat']:.4f}",
                    "Status":        "DRIFT ⚠" if r["is_drifted"] else "OK ✓",
                    "Baseline Mean": f"{r.get('baseline_mean', 'N/A')}",
                    "Recent Mean":   f"{r.get('recent_mean', 'N/A')}",
                    "Samples":       str(r.get("recent_n", "-")),
                })

        df_drift = pd.DataFrame(rows)

        def color_status(val):
            if "DRIFT" in str(val):
                return "color: #ff4444; font-weight: bold"
            if "OK" in str(val):
                return "color: #00b050; font-weight: bold"
            return ""

        styled_drift = df_drift.style.applymap(
            color_status, subset=["Status"]
        )
        st.dataframe(
            styled_drift,
            width="stretch",
            hide_index=True
        )

        # Drift explanation
        with st.expander("📖 How drift detection works"):
            st.markdown("""
            **KS-test (Kolmogorov-Smirnov test)**

            - Compares the full AQI distribution shape, not just means
            - Baseline: 3-year annual distribution (2022-2025)
            - Recent window: last 7 days of live data
            - p-value < 0.05 = distributions are statistically different

            **Why cities show drift in April:**
            April data differs from the annual baseline because the
            baseline includes winter pollution spikes (Oct-Feb) and
            monsoon dips (Jun-Sep). April summer data has a narrower
            distribution — KS-test correctly detects this seasonal shift.

            **Production improvement:**
            Use same-season baseline (compare April vs April from
            previous years) to distinguish seasonal drift from
            genuine distribution shift.
            """)

    # ── MLflow links ──────────────────────────────────────────────────────
    st.subheader("MLflow Experiment Tracking")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>MLflow UI</strong><br>
            View experiments, runs, metrics, and artifacts<br>
            <a href="http://localhost:5000" target="_blank">
                → Open MLflow at localhost:5000
            </a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>Model Registry</strong><br>
            Champion models use @champion alias (MLflow 3.x)<br>
            Load: <code>mlflow.prophet.load_model(
            'models:/aqi_prophet_delhi@champion')</code>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Data Pipeline
# ══════════════════════════════════════════════════════════════════════════

elif page == "📊 Data Pipeline":

    st.title("📊 Data Pipeline Status")
    st.caption(
        "Airflow DAGs run hourly to ingest AQI + weather data. "
        "Pipeline status based on latest raw data files."
    )

    # ── Data sources ──────────────────────────────────────────────────────
    st.subheader("Data Sources")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Training Data</p>
            <p style="color:#4a9eff;font-size:1rem;font-weight:bold">
                Kaggle Dataset
            </p>
            <p style="color:#666;font-size:0.8rem">
                2022-08-05 → 2025-11-26<br>
                ~29,000 hours per city<br>
                5 Indian cities
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Live AQI</p>
            <p style="color:#4a9eff;font-size:1rem;font-weight:bold">
                OpenAQ v3 API
            </p>
            <p style="color:#666;font-size:0.8rem">
                Hourly ingestion<br>
                PM2.5 → India CPCB AQI<br>
                CPCB stations
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Weather + Inference</p>
            <p style="color:#4a9eff;font-size:1rem;font-weight:bold">
                OpenMeteo + AQICN
            </p>
            <p style="color:#666;font-size:0.8rem">
                OpenMeteo: hourly forecast<br>
                AQICN: live AQI (current hour)<br>
                Free APIs, no key for OpenMeteo
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline architecture ──────────────────────────────────────────────
    st.subheader("Pipeline Architecture")
    st.code("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    DATA INGESTION LAYER                     │
    │                                                             │
    │  Kaggle Dataset ──────────────────────────────────────────► │
    │  (2022-2025, seed data)                                     │
    │                                                             │
    │  OpenAQ API ──► aqi_hourly_ingest DAG ──► data/raw/{city}/  │
    │  OpenMeteo  ──► (runs every hour)      ──► date=*.parquet   │
    │  AQICN      ──► (drift check)                               │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  FEATURE ENGINEERING LAYER                  │
    │                                                             │
    │  pipeline.py (PySpark + Pandas)                             │
    │  → Lag features (1h, 24h, 168h)                             │
    │  → Rolling statistics (6h, 24h, 7d)                         │
    │  → Cyclical encodings (hour, day, month)                    │
    │  → Calendar features (rush hour, crop burning)              │
    │  → Output: data/processed/{city}/features.parquet           │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    MODEL TRAINING LAYER                     │
    │                                                             │
    │  trainer.py                                                 │
    │  → Prophet (primary) + SARIMA (baseline)                    │
    │  → MLflow experiment tracking                               │
    │  → Champion promotion via @champion alias                   │
    │  → Triggered by: drift detection or manual                  │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     SERVING LAYER                           │
    │                                                             │
    │  FastAPI (port 8000)                                        │
    │  → Loads champion models from MLflow registry               │
    │  → /forecast/{city} → 24h AQI prediction                    │
    │  → /drift → KS-test drift status                            │
    │  → /latest/{city} → current AQI from AQICN                  │
    │  → Prometheus metrics at /metrics                           │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   MONITORING LAYER                          │
    │                                                             │
    │  Prometheus (port 9090) ──► scrapes /metrics every 15s      │
    │  Grafana    (port 3001) ──► visualises request rate,        │
    │                             latency, error rate             │
    │  Airflow    (port 8080) ──► DAG monitoring, task logs       │
    │  MLflow     (port 5000) ──► experiment tracking UI          │
    └─────────────────────────────────────────────────────────────┘
    """, language=None)

    # ── Airflow links ──────────────────────────────────────────────────────
    st.subheader("Pipeline Management")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>Airflow UI</strong><br>
            Monitor DAG runs, task logs, and trigger backfills<br>
            <a href="http://localhost:8080" target="_blank">
                → Open Airflow at localhost:8080
            </a><br>
            <small>Login: admin / admin</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>Prometheus + Grafana</strong><br>
            Real-time API metrics and alerting<br>
            <a href="http://localhost:3001" target="_blank">
                → Open Grafana at localhost:3001
            </a><br>
            <a href="http://localhost:9090" target="_blank">
                → Open Prometheus at localhost:9090
            </a>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature engineering ────────────────────────────────────────────────
    st.subheader("Feature Engineering")
    with st.expander("View feature set used for training"):
        st.markdown("""
        | Feature | Description | Source |
        |---------|-------------|--------|
        | `y` (target) | India CPCB AQI | OpenAQ/Kaggle |
        | `aqi_lag_1h` | AQI 1 hour ago | Computed |
        | `aqi_lag_24h` | AQI 24 hours ago | Computed |
        | `aqi_lag_168h` | AQI 7 days ago | Computed |
        | `aqi_roll_mean_24h` | 24h rolling mean | Computed |
        | `aqi_roll_mean_7d` | 7-day rolling mean | Computed |
        | `temperature_2m` | Temperature °C | OpenMeteo |
        | `wind_speed_10m` | Wind speed m/s | OpenMeteo |
        | `relative_humidity_2m` | Humidity % | OpenMeteo |
        | `is_rush_hour` | Morning/evening rush | Calendar |
        | `crop_burning` | Oct-Nov season | Calendar |
        | `festival_period` | Oct-Nov festivals | Calendar |
        | `hour_sin/cos` | Cyclical hour encoding | Calendar |
        | `dow_sin/cos` | Cyclical day encoding | Calendar |
        """)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ══════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":

    st.title("ℹ️ About AQI Prediction System")

    st.markdown("""
    ## What is this system?

    The **AQI Prediction System** forecasts air quality for 5 major
    Indian cities — Delhi, Mumbai, Kolkata, Chennai, and Bengaluru —
    24 hours in advance using machine learning.

    ## How to use (Non-technical guide)

    1. **Select a city** from the dropdown on the Forecast page
    2. **Read the current AQI** — the large number shows today's
       air quality
    3. **Check the forecast chart** — the blue line shows predicted
       AQI for the next 24 hours. The shaded area shows the
       uncertainty range.
    4. **Follow the health advisory** — shown below the current AQI

    ## AQI Scale (India CPCB)
    """)

    aqi_scale = pd.DataFrame([
        {"Category": "Good",         "Range": "0-50",   "Color": "🟢",
         "Advisory": "Air quality is good. Ideal for outdoor activities."},
        {"Category": "Satisfactory", "Range": "51-100", "Color": "🟡",
         "Advisory": "Minor discomfort for sensitive people."},
        {"Category": "Moderate",     "Range": "101-200","Color": "🟡",
         "Advisory": "Breathing discomfort for people with lung/heart disease."},
        {"Category": "Poor",         "Range": "201-300","Color": "🟠",
         "Advisory": "Breathing discomfort for most people."},
        {"Category": "Very Poor",    "Range": "301-400","Color": "🔴",
         "Advisory": "Respiratory illness on prolonged exposure."},
        {"Category": "Severe",       "Range": "401-500","Color": "🔴",
         "Advisory": "Health impact even on light physical activity."},
    ])
    st.dataframe(aqi_scale, width="stretch", hide_index=True)

    st.markdown("""
    ## Technology Stack

    | Component | Technology |
    |-----------|------------|
    | Forecast Model | Facebook Prophet + SARIMA |
    | Data Engineering | Apache Spark + Airflow |
    | Experiment Tracking | MLflow 3.x |
    | API | FastAPI + Uvicorn |
    | Monitoring | Prometheus + Grafana |
    | Containerization | Docker Compose |
    | Data Sources | Kaggle, OpenAQ, OpenMeteo, AQICN |

    ## Model Information

    - **Primary model:** Facebook Prophet with weather regressors
    - **Baseline model:** SARIMA (1,1,1)(1,1,1,24)
    - **Training data:** 3.5 years (Aug 2022 → Apr 2026)
    - **Test MAE:** 19-31 AQI points depending on city
    - **Forecast horizon:** 24 hours
    - **Retraining:** Automatic when KS-test detects distribution drift
    """)

    with st.expander("📋 User Manual"):
        st.markdown("""
        ### Quick Start

        1. Open browser at **http://localhost:8501**
        2. Select city from sidebar dropdown
        3. View current AQI and 24h forecast

        ### Interpreting the Forecast

        - **Blue line** = predicted AQI
        - **Shaded area** = 95% confidence interval
        - **Dotted lines** = AQI category thresholds

        ### What the numbers mean

        AQI below 100 = safe for most people
        AQI 100-200   = sensitive groups should limit outdoor activity
        AQI above 200 = everyone should reduce outdoor exposure
        AQI above 300 = avoid outdoor activity

        ### Data freshness

        - Current AQI updates every hour via AQICN
        - Forecast regenerated on page load (cached 5 minutes)
        - Models retrain automatically when data drift is detected
        """)
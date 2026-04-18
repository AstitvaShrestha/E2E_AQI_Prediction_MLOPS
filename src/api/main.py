"""
src/api/main.py

FastAPI inference API for AQI forecasting.
Endpoints:
  GET /health              — health check
  GET /forecast/{city}     — 24h AQI forecast
  GET /drift               — drift status for all cities
  GET /cities              — list available cities
  GET /latest/{city}       — latest AQI reading
  POST /retrain/{city}     — trigger retraining (Airflow webhook)

Loads champion Prophet models from MLflow registry at startup.
Falls back to disk if MLflow unavailable.
"""

import os
import sys
import logging
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.prophet
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

try:
    from ..config import CITIES, get_aqi_category, pm25_to_aqi
    from ..ingest.aqicn import get_current_aqi_value
    from ..ingest.weather import get_weather_forecast
    from ..features.drift import check_all_cities
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES, get_aqi_category, pm25_to_aqi
    from src.ingest.aqicn import get_current_aqi_value
    from src.ingest.weather import get_weather_forecast
    from src.features.drift import check_all_cities

logger     = logging.getLogger(__name__)
DATA_DIR   = Path(os.getenv("DATA_DIR",   "data"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — runs before app starts accepting requests
    load_champion_models()
    logger.info("Models loaded — API ready")
    yield
    # Shutdown — runs when app is stopping
    champion_models.clear()
    logger.info("Models cleared — API shutdown")

# ----App Setup------

app = FastAPI(
    title="AQI Prediction API",
    description="24-hour AQI forecasting for 5 Indian cities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

#----Model Registry------
champion_models = {}

def load_champion_models():
    """
    Load champion Prophet models from MLflow registry at startup.
    Falls back gracefully if a city model is missing.
    """

    mlflow.set_tracking_uri(MLFLOW_URI)

    for city in CITIES:
        model_name = f"aqi_prophet_{city.lower().replace(' ', '_')}"

        try:
            model_uri = f"models:/{model_name}@champion"
            model = mlflow.prophet.load_model(model_uri)
            champion_models[city] = model
            logger.info(f"Loaded champion model for {city}")

        except Exception as e:
            logger.error(
                f"Failed to load champion model for {city}: {e}"
            )

    logger.info(
        f"Loaded {len(champion_models)}/5 champion models"
    )

# @app.on_event("startup")
# async def startup_event():
#     load_champion_models()


#----Regressor helpers-------

def _get_recent_weather_means(city):
    """Mean weather values from last 7 days — fallback for missing hours."""
    
    city_key = city.lower().replace(" ", "_")
    files = sorted(
        glob.glob(f"{DATA_DIR}/raw/{city_key}/weather/date=*.parquet"),
        reverse=True
    )[:7]

    if not files:
        return {}
    

    try:
        df = pd.concat([pd.read_parquet(f) for f in files])
        return {
            col: float(df[col].mean())
            for col in [
                "temperature_2m", "wind_speed_10m",
                "relative_humidity_2m", "precipitation",
                "surface_pressure", "wind_direction_10m",
            ]
            if col in df.columns and df[col].notna().any()
        }
    
    except Exception as e:
        logger.error(f"Error calculating recent weather means for {city}: {e}")
        return {}
    

def _get_recent_aqi_mean(city):
    """Mean AQI from last 7 days — fallback for missing hours."""
    
    city_key = city.lower().replace(" ", "_")
    files    = sorted(
        glob.glob(f"{DATA_DIR}/raw/{city_key}/date=*.parquet"),
        reverse=True
    )[:7]

    if not files:
        return 100.0

    try:
        df = pd.concat([pd.read_parquet(f) for f in files])
        return float(df["aqi"].mean())
    
    except Exception as e:
        return 100.0


def _get_last_known_aqi(city):
    """Last known AQI from disk — fallback if AQICN unavailable."""

    city_key = city.lower().replace(" ", "_")
    files    = sorted(
        glob.glob(f"{DATA_DIR}/raw/{city_key}/date=*.parquet"),
        reverse=True
    )

    for f in files[:7]:
        try:
            df = pd.read_parquet(f)
            if not df.empty and "aqi" in df.columns:
                return float(
                    df.sort_values("timestamp").iloc[-1]["aqi"]
                )
            
        except Exception as e:
            logger.error(f"Error reading AQI data for {city}: {e}")
            continue
    
    return None

def build_future_dataframe(city, model, horizon=24):
    """
    Build future dataframe for Prophet inference.

    Fallback chain per regressor:
      weather -> OpenMeteo forecast -> recent historical mean
      aqi_lag_* -> AQICN current -> last known from disk -> 100
      aqi_roll_* -> recent raw mean
      calendar -> computed from timestamps (always available)
      bonus cols -> 0 (conservative default)
    """

    # Future timestamps start from now
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    future = pd.DataFrame({
        "ds": pd.date_range(start=now, periods=horizon, freq="h")
    })


    # Weather regressors
    WEATHER_COLS = [
        "temperature_2m", "wind_speed_10m",
        "wind_direction_10m", "relative_humidity_2m",
        "precipitation", "surface_pressure",
    ]

    try:
        wx_df = get_weather_forecast(city, periods=horizon)
        
        if not wx_df.empty:
            wx_df["timestamp"] = (
                pd.to_datetime(wx_df["timestamp"], utc=True)
                .dt.tz_localize(None)
                .dt.floor("h")
            )

            wx_map = wx_df.set_index("timestamp")

            for col in WEATHER_COLS:

                if col in wx_map.columns:
                    future[col] = future["ds"].map(wx_map[col])
                
    except Exception as e:
        logger.warning(f"Weather forecast failed for {city}: {e}")

    # Fill missing weather with recent historical mean
    recent_means = _get_recent_weather_means(city)

    for col in WEATHER_COLS:
        if col not in future.columns:
            future[col] = recent_means.get(col, 0.0)
        else:
            future[col].fillna(recent_means.get(col, 0.0), inplace=True)

    # AQI lag regressors
    # Primary: AQICN live
    # Fallback: last known from disk
    # Last resort: city recent mean

    current_aqi = get_current_aqi_value(city)

    if current_aqi is None:
        current_aqi = _get_last_known_aqi(city)

    if current_aqi is None:
        current_aqi = _get_recent_aqi_mean(city)

    recent_mean = _get_recent_aqi_mean(city)

    for reg in model.extra_regressors.keys():
        if "lag" in reg:
            future[reg] = current_aqi
        elif "roll" in reg:
            future[reg] = recent_mean

    # Calendar regressors
    
    if "is_rush_hour" in model.extra_regressors:
        future["is_rush_hour"] = (
            future["ds"].dt.hour.between(7, 10) |
            future["ds"].dt.hour.between(17, 20)
        ).astype(int)

    
    # Bonus regressors

    # for reg in ["temp_inversion", "crop_burning", "festival_period"]:
    #     if reg in model.extra_regressors:
    #         future[reg] = 0

    if "crop_burning" in model.extra_regressors:
        future["crop_burning"] = (
            future["ds"].dt.month.isin([10, 11])
        ).astype(float)

    if "festival_period" in model.extra_regressors:
        future["festival_period"] = future["ds"].dt.month.map({
            10: 0.55, 11: 0.50
        }).fillna(0.0)

    if "temp_inversion" in model.extra_regressors:
        future["temp_inversion"] = 0.0  # always 0 — confirmed from data
        
    # Safety net - fill any remaining NaN
    # Prophet crashes on NaN regressors

    for reg in model.extra_regressors.keys():
        if reg not in future.columns:
            future[reg] = 0.0
        else:
            future[reg].fillna(0.0, inplace=True)

    return future


# ------Endpoints-------

@app.get("/health")
async def health():
    """Health check — returns loaded models and system status."""
    
    return {
        "status": "healthy",
        "models_loaded":  list(champion_models.keys()),
        "models_missing": [
            c for c in CITIES if c not in champion_models
        ],
        "mlflow_uri":     MLFLOW_URI,
        "timestamp":      datetime.utcnow().isoformat(),
    }


@app.get("/cities")
async def list_cities():
    """List available cities with their configuration."""
    
    return {
        city: {
            "lat":        cfg["lat"],
            "lon":        cfg["lon"],
            "timezone":   cfg["timezone"],
            "model_loaded": city in champion_models,
        }

        for city, cfg in CITIES.items()
    }


@app.get("/forecast/{city}")
async def forecast(city: str, horizon: int=24):
    """
    Generate 24-hour AQI forecast for a city.

    Returns:
      hourly AQI predictions with confidence intervals
      AQI category labels (Good/Moderate/Poor etc)
      data source information
    """

    # Validate city
    if city not in CITIES:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found. "
                   f"Available: {list(CITIES.keys())}"
        )
    
    if city not in champion_models:
        raise HTTPException(
            status_code=503,
            detail=f"No champion model loaded for {city}. "
                   "Run trainer.py first."
        )
    
    model = champion_models[city]

    try:
        # Build future dataframe starting from NOW
        future   = build_future_dataframe(city, model, horizon=horizon)
        forecast = model.predict(future)

        # Clip to valid AQI range
        forecast["yhat"]  = np.clip(forecast["yhat"], 0, 500)
        forecast["yhat_lower"] = np.clip(forecast["yhat_lower"], 0, 500)
        forecast["yhat_upper"] = np.clip(forecast["yhat_upper"], 0, 500)

        from datetime import timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))

        # Build response
        predictions = []

        for _, row in forecast.iterrows():
            aqi = int(round(row["yhat"]))
            category = get_aqi_category(aqi)
            predictions.append({
                "timestamp":   row["ds"].isoformat(), # UTC
                "timestamp_ist": row["ds"].replace(               # IST
                    tzinfo=timezone.utc
                ).astimezone(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                "aqi":         aqi,
                "aqi_lower":   int(round(row["yhat_lower"])),
                "aqi_upper":   int(round(row["yhat_upper"])),
                "category":    category,
            })

        # Get current AQI for context
        current_aqi = get_current_aqi_value(city)

        return {
            "city":          city,
            "generated_at":  datetime.utcnow().isoformat(),
            "horizon_hours": horizon,
            "current_aqi":   current_aqi,
            "forecast":      predictions,
            "model": {
                "name":    f"aqi_prophet_{city.lower().replace(' ','_')}",
                "alias":   "champion",
            }
        }
    
    except Exception as e:
        logger.error(f"Forecast failed for {city}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(e)}"
        )
    

@app.get("/latest/{city}")
async def latest_aqi(city: str):
    """
    Get latest AQI reading for a city.
    Primary: AQICN live API
    Fallback: last known from disk
    """

    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")

    
    aqi = get_current_aqi_value(city)

    if aqi is None:
        raise HTTPException(
            status_code=503,
            detail=f"No AQI data available for {city}"
        )
    
    return {
        "city":      city,
        "aqi":       aqi,
        "category":  get_aqi_category(int(aqi)),
        "timestamp": datetime.utcnow().isoformat(),
        "source":    "aqicn",
    }

@app.get("/drift")
async def drift_status():
    """
    Run KS-test drift detection for all cities.
    Returns drift status, p-values, and retraining recommendations.
    """
     
    try:
        results = check_all_cities()
        cities_drifted = [
            city for city, r in results.items()
            if r.get("is_drifted") and not r.get("error")
        ]

        return {
            "checked_at":       datetime.utcnow().isoformat(),
            "cities_drifted":   cities_drifted,
            "retrain_required": len(cities_drifted) > 0,
            "results":          results,
        }
    
    except Exception as e:
        logger.error(f"Drift status check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Drift check failed: {str(e)}"
        )
    

@app.post("/retrain/{city}")
async def trigger_retrain(city: str, background_tasks: BackgroundTasks):
    """
    Trigger model retraining for a city.
    Runs in background — returns immediately.
    Called by Airflow DAG when drift is detected.
    """
     
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")

    def retrain_task(city: str):
        """Background retraining task."""
        
        import subprocess
        logger.info(f"Retraining triggered for {city}")
        try:
            result = subprocess.run(
                [
                    "python", "-c",
                    f"""
                        import sys
                        from pathlib import Path
                        sys.path.insert(0, str(Path(r\"{Path(__file__).resolve().parents[2]}\")))
                        from src.train.trainer import train_prophet, evaluate_and_promote
                        import pandas as pd

                        city = '{city}'
                        feat_path = Path('data/processed/{city.lower().replace(" ","_")}/features.parquet')
                        df = pd.read_parquet(feat_path)
                        model, metrics = train_prophet(city, df)
                        promoted = evaluate_and_promote(city, metrics['mae'], metrics['version'])
                        print(f'Retrain complete: MAE={{metrics[\"mae\"]:.2f}} promoted={{promoted}}')
                    """
                ],
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.returncode == 0:
                logger.info(f"Retrain complete for {city}: {result.stdout}")

                # Reload champion model after retraining
                model_name = f"aqi_prophet_{city.lower().replace(' ','_')}"
                model_uri  = f"models:/{model_name}@champion"
                champion_models[city] = mlflow.prophet.load_model(model_uri)
                logger.info(f"Reloaded champion model for {city}")
            
            else:
                logger.error(f"Retrain failed for {city}: {result.stderr}")
        
        except Exception as e:
            logger.error(f"Retrain task failed for {city}: {e}")

    background_tasks.add_task(retrain_task, city)

    return {
        "status":  "retraining_started",
        "city":    city,
        "message": f"Retraining triggered for {city} in background",
    }


@app.post("/reload-models")
async def reload_models():
    """
    Reload champion models from MLflow registry.
    Call this after retraining to pick up new champion models.
    """

    load_champion_models()

    return {
        "status":        "reloaded",
        "models_loaded": list(champion_models.keys()),
        "timestamp":     datetime.utcnow().isoformat(),
    }


#---Entry point-----

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
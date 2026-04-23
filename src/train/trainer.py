"""
Model training — Prophet (primary) + SARIMA (baseline).
Full MLflow experiment tracking with custom metrics and artifacts.
Logs all experiments to MLflow with full tracking:
  - Parameters
  - Metrics (MAE, RMSE, MAPE)
  - Artifacts (forecast plots, residuals)
  - Feature importance
  - Model registry with Production promotion

Promotion logic: new model replaces production only if MAE improves.
"""

import os
from pyexpat import model
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

import mlflow
import mlflow.prophet
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


try:
    from ..config import CITIES, get_aqi_category
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES, get_aqi_category

from dotenv import load_dotenv
load_dotenv()


logger     = logging.getLogger(__name__)
DATA_DIR   = Path(os.getenv("DATA_DIR",   "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


REGRESSORS = [
    "wind_speed_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "aqi_lag_24h",
    "aqi_roll_mean_24h",
    "is_rush_hour",
]
BONUS_REGRESSORS = [
    "temp_inversion",
    "crop_burning",
    "festival_period",
]

FORECAST_HORIZON = 24   # hours ahead to predict

def get_or_create_experiment(city):
    """Get or create MLflow experiment for a city."""

    name = f"aqi_forecast_{city.lower().replace(' ', '_')}"
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(name)

    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info(f"Created new MLflow experiment '{name}' with ID {exp_id}")
    
    else:
        exp_id = exp.experiment_id
        logger.info(f"Using existing MLflow experiment '{name}' with ID {exp_id}")

    return exp_id


def get_champion_mae(city):
    """
    Get MAE of current champion model.
    MLflow 3.x uses aliases instead of stages.
    champion alias = Production in MLflow 2.x
    """

    client = mlflow.MlflowClient()
    model_name = f"aqi_prophet_{city.lower().replace(' ', '_')}"

    try:

        version = client.get_model_version_by_alias(model_name, "champion")
        run = client.get_run(version.run_id)
        return float(run.data.metrics.get("mae", float("inf")))
    
    except Exception:
        return float("inf")
    

# ----Train / Test split -----------

def split_train_test(df, test_days=7):

    cutoff = df["ds"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["ds"] <= cutoff].dropna(subset=["y"]).copy()
    test_df  = df[df["ds"] >  cutoff].dropna(subset=["y"]).copy()

    # Prophet requires timezone-naive timestamps
    train_df["ds"] = train_df["ds"].dt.tz_localize(None)
    test_df["ds"]  = test_df["ds"].dt.tz_localize(None)

    return train_df, test_df


# -----Plots-----

def plot_forecast(model, forecast, city, path):
    fig = model.plot(forecast)
    fig.suptitle(f"{city} — Prophet AQI Forecast", fontsize=12)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true, y_pred, city, path):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, alpha=0.7, color="#378ADD", linewidth=0.8)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"{city} — Residuals")
    axes[1].hist(residuals, bins=40, color="#378ADD", alpha=0.7)
    axes[1].set_title(f"{city} — Residual distribution")
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_components(model, forecast, city, path):
    fig = model.plot_components(forecast)
    fig.suptitle(f"{city} — Seasonality Components", fontsize=12)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ----SARIMA baseline------------

def train_sarima(city, df):
    """Train SARIMA as statistical baseline — logged to MLflow."""

    exp_id = get_or_create_experiment(city)
    run_name = f"sarima_{city.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    train_df, test_df = split_train_test(df)

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.log_params({
            "model_type": "SARIMA",
            "order": "(1,1,1)",
            "seasonal_order": "(1,1,1,24)",
            "city": city,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        })

        try:
            model = SARIMAX(
                train_df["y"].values,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)

            y_true   = test_df["y"].values
            y_pred   = np.array(model.forecast(steps=len(test_df)))
            mae      = float(mean_absolute_error(y_true, y_pred))
            rmse     = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mape     = float(np.mean(
                np.abs((y_true - y_pred) / (y_true + 1e-8))
            ) * 100)

            mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
            logger.info(f"SARIMA {city}: MAE={mae:.2f}")
            return {"mae": mae, "rmse": rmse, "mape": mape}
        
        except Exception as e:
            logger.error(f"SARIMA failed for {city}: {e}")
            mlflow.log_metric("mae", 9999)
            return {"mae": 9999, "rmse": 9999, "mape": 9999}
        

# ---- Prophet model training -----------

def train_prophet(city, df, run_name=None):
    """Train Prophet model with regressors, logged to MLflow."""

    exp_id = get_or_create_experiment(city)
    run_name = run_name or (f"prophet_{city.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}")

    train_df, test_df = split_train_test(df)

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:

        # log parameters
        mlflow.log_params({
            "model_type": "Prophet",
            "city": city,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "quarterly_seasonality": True,
            "quarterly_fourier_order": 5,
            "hourly_fourier_order": 8,
            "forecast_horizon_h": FORECAST_HORIZON,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        })

        # Build model
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )

        model.add_seasonality(
            name="quarterly",
            period=365.25 / 4,   # 91.3 days
            fourier_order=5,      # complexity of the pattern
        )

        model.add_seasonality(name="hourly_pattern", period=1, fourier_order=8)
      

        # Add regressors

        available = [
            r for r in REGRESSORS
            if r in train_df.columns and train_df[r].notna().sum() > 0
        ]

        for r in BONUS_REGRESSORS:
            if r in train_df.columns and \
               train_df[r].notna().sum() > 0:
                available.append(r)

        for reg in available:
            model.add_regressor(reg)
            train_df[reg] = train_df[reg].fillna(0.0)

        # Train
        logger.info(
            f"Training Prophet {city}: "
            f"{len(train_df)} rows, "
            f"{len(available)} regressors"
        )
        model.fit(train_df)

        # Predict
        # future = model.make_future_dataframe(
        #     periods=len(test_df), freq="H", include_history=False
        # )

        future = test_df[["ds"]].copy().reset_index(drop=True)
        
        for reg in available:
            if reg in test_df.columns:
                future[reg] = future["ds"].map(
                    test_df.set_index("ds")[reg]
                ).fillna(train_df[reg].mean())

            else:
                future[reg] = train_df[reg].mean() \
                              if reg in train_df.columns else 0
                
        forecast = model.predict(future)
        y_true   = test_df["y"].values
        y_pred   = np.clip(
            forecast["yhat"].values[:len(y_true)], 0, 500
        )

        # Metrics
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(
            np.mean(np.abs((y_true - y_pred) /
                           (y_true + 1e-8))) * 100
        )


        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        logger.info(
            f"Prophet {city}: MAE={mae:.2f} "
            f"RMSE={rmse:.2f} MAPE={mape:.1f}%"
        )

        # Artifacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MODELS_DIR / "tmp"
        tmp.mkdir(exist_ok=True)

        plot_forecast(model, forecast, city,
                      str(tmp / "forecast.png"))
        mlflow.log_artifact(str(tmp / "forecast.png"))

        plot_residuals(y_true, y_pred, city,
                       str(tmp / "residuals.png"))
        mlflow.log_artifact(str(tmp / "residuals.png"))

        plot_components(model, forecast, city,
                        str(tmp / "components.png"))
        mlflow.log_artifact(str(tmp / "components.png"))

        # Model logging
        model_info = mlflow.prophet.log_model(
            model,
            name="model",        
        )

        # Register model using URI from log_model result
        model_name = f"aqi_prophet_{city.lower().replace(' ', '_')}"
        registered = mlflow.register_model(
            model_uri=model_info.model_uri,  # MLflow 3.x: use model_uri
            name=model_name,
        )

        logger.info(
            f"Registered {model_name} v{registered.version}"
        )

        return model, {"mae": mae, "rmse": rmse, "mape": mape,
                       "version": registered.version}
    

def evaluate_and_promote(city, new_mae, new_version):
    """Promote to champion if MAE improves."""
    
    client = mlflow.MlflowClient()
    model_name = f"aqi_prophet_{city.lower().replace(' ', '_')}"
    champion_mae = get_champion_mae(city)

    if new_mae < champion_mae:
        # Remove champion alias from old version
        try:
            old = client.get_model_version_by_alias(model_name, "champion")
            client.delete_registered_model_alias(model_name, "champion")
            
            # Tag old version as retired
            client.set_model_version_tag(
                model_name, old.version, "status", "retired"
            )
            logger.info(
                f"{city}: retired old champion v{old.version}"
            )

        except Exception:
            pass  # no existing champion — first time

        # Set new version as champion
        client.set_registered_model_alias(
            model_name, "champion", str(new_version)
        )

        # Tag with metadata
        client.set_model_version_tag(
            model_name, str(new_version), "status", "champion"
        )

        client.set_model_version_tag(
            model_name, str(new_version),
            "mae", str(round(new_mae, 4))
        )
        client.set_model_version_tag(
            model_name, str(new_version),
            "promoted_at", datetime.now().isoformat()
        )

        logger.info(
            f"{city}: promoted v{new_version} as champion "
            f"(MAE {new_mae:.2f} < {champion_mae:.2f})"
        )

        return True
    
    else:
        # Tag as challenger (not champion)
        client.set_model_version_tag(
            model_name, str(new_version),
            "status", "challenger"
        )
        client.set_model_version_tag(
            model_name, str(new_version),
            "mae", str(round(new_mae, 4))
        )

        logger.info(
            f"{city}: v{new_version} tagged as challenger "
            f"(MAE {new_mae:.2f} >= {champion_mae:.2f})"
        )

        return False
    

# ---- Main ---------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f"MLflow: {MLFLOW_URI}")
    print(f"MLflow version: {mlflow.__version__}\n")

    summary = {}

    for city in CITIES:
        print(f"\n{'='*50}")
        print(f"  {city}")
        print(f"{'='*50}")

        feat_path = (
            DATA_DIR / "processed" /
            city.lower().replace(" ", "_") /
            "features.parquet"
        )

        if not feat_path.exists():
            print(f"  ERROR: run pipeline.py first")
            continue

        df = pd.read_parquet(feat_path)
        print(f"  Rows: {len(df):,} | "
              f"{df['ds'].min().date()} → {df['ds'].max().date()}")

        # SARIMA baseline
        print(f"  Training SARIMA...")
        sarima_m = train_sarima(city, df)
        print(f"  SARIMA  MAE: {sarima_m['mae']:.2f}")

        # Prophet
        print(f"  Training Prophet...")
        model, prophet_m = train_prophet(city, df)
        print(f"  Prophet MAE: {prophet_m['mae']:.2f}")

        # Promote if better
        promoted = evaluate_and_promote(
            city,
            prophet_m["mae"],
            prophet_m["version"],
        )
        status = "champion ✓" if promoted else "challenger"
        print(f"  Status: {status}")

        summary[city] = {
            "sarima_mae":  sarima_m["mae"],
            "prophet_mae": prophet_m["mae"],
            "improvement": sarima_m["mae"] - prophet_m["mae"],
            "promoted":    promoted,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"{'City':<12} {'SARIMA':<10} {'Prophet':<10} "
          f"{'Improvement':<14} {'Status'}")
    print(f"{'-'*60}")
    for city, m in summary.items():
        status = "champion" if m["promoted"] else "challenger"
        print(
            f"{city:<12} "
            f"{m['sarima_mae']:<10.2f} "
            f"{m['prophet_mae']:<10.2f} "
            f"{m['improvement']:<14.2f} "
            f"{status}"
        )

    print(f"\nMLflow UI: {MLFLOW_URI}")
    print("Load champion model:")
    print("  mlflow.prophet.load_model("
          "'models:/aqi_prophet_delhi@champion')")
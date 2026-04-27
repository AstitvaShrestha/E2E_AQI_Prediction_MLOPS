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


def _compute_regression_metrics(y_true, y_pred):
    """Compute MAE, RMSE, and MAPE for numeric predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _compute_category_accuracy(y_true, y_pred):
    """Compute AQI category match accuracy (%) using CPCB category labels."""
    true_labels = [get_aqi_category(int(v))["label"] for v in y_true]
    pred_labels = [get_aqi_category(int(v))["label"] for v in y_pred]
    if not true_labels:
        return 0.0
    return float(np.mean(np.array(true_labels) == np.array(pred_labels)) * 100)


def _compute_coverage_95(y_true, y_lower, y_upper):
    """Compute empirical 95% interval coverage (%)."""
    if len(y_true) == 0:
        return 0.0
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(covered) * 100)


def _compute_boundary_error_rate(y_true, y_pred, margin=10):
    """% of points that are category-mismatched near AQI boundaries."""
    if len(y_true) == 0:
        return 0.0
    boundaries = np.array([50, 100, 200, 300, 400], dtype=float)
    true_labels = np.array([get_aqi_category(int(v))["label"] for v in y_true])
    pred_labels = np.array([get_aqi_category(int(v))["label"] for v in y_pred])
    mismatches = true_labels != pred_labels
    dist_to_boundary = np.min(np.abs(y_true.reshape(-1, 1) - boundaries), axis=1)
    near_boundary = dist_to_boundary <= margin
    return float(np.mean(mismatches & near_boundary) * 100)


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

            y_train_true = train_df["y"].values
            y_train_pred = np.array(model.fittedvalues)
            train_metrics = _compute_regression_metrics(y_train_true, y_train_pred)

            y_test_true = test_df["y"].values
            forecast_res = model.get_forecast(steps=len(test_df))
            y_test_pred = np.array(forecast_res.predicted_mean)
            conf_int = forecast_res.conf_int(alpha=0.05)
            y_test_lower = np.array(conf_int.iloc[:, 0])
            y_test_upper = np.array(conf_int.iloc[:, 1])
            test_metrics = _compute_regression_metrics(y_test_true, y_test_pred)
            category_accuracy = _compute_category_accuracy(y_test_true, y_test_pred)
            coverage_95 = _compute_coverage_95(y_test_true, y_test_lower, y_test_upper)
            boundary_error_rate = _compute_boundary_error_rate(y_test_true, y_test_pred)

            gap_metrics = {
                "gap_mae": test_metrics["mae"] - train_metrics["mae"],
                "gap_rmse": test_metrics["rmse"] - train_metrics["rmse"],
                "gap_mape": test_metrics["mape"] - train_metrics["mape"],
            }

            mlflow.log_metrics({
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "mape": test_metrics["mape"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_mape": train_metrics["mape"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_mape": test_metrics["mape"],
                "category_accuracy": category_accuracy,
                "coverage_95": coverage_95,
                "boundary_error_rate": boundary_error_rate,
                "test_category_accuracy": category_accuracy,
                "test_coverage_95": coverage_95,
                "test_boundary_error_rate": boundary_error_rate,
                **gap_metrics,
            })
            logger.info(
                f"SARIMA {city}: test_MAE={test_metrics['mae']:.2f} "
                f"train_MAE={train_metrics['mae']:.2f} "
                f"gap_MAE={gap_metrics['gap_mae']:.2f} "
                f"CatAcc={category_accuracy:.1f}% "
                f"Coverage95={coverage_95:.1f}% "
                f"BoundaryErr={boundary_error_rate:.1f}%"
            )
            return {
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "mape": test_metrics["mape"],
                "category_accuracy": category_accuracy,
                "coverage_95": coverage_95,
                "boundary_error_rate": boundary_error_rate,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_mape": train_metrics["mape"],
                **gap_metrics,
            }
        
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
        y_test_true = test_df["y"].values
        y_test_pred = np.clip(
            forecast["yhat"].values[:len(y_test_true)], 0, 500
        )
        y_test_lower = np.clip(
            forecast["yhat_lower"].values[:len(y_test_true)], 0, 500
        )
        y_test_upper = np.clip(
            forecast["yhat_upper"].values[:len(y_test_true)], 0, 500
        )

        train_pred_df = model.predict(train_df[["ds"] + available])
        y_train_true = train_df["y"].values
        y_train_pred = np.clip(
            train_pred_df["yhat"].values[:len(y_train_true)], 0, 500
        )

        train_metrics = _compute_regression_metrics(y_train_true, y_train_pred)
        test_metrics = _compute_regression_metrics(y_test_true, y_test_pred)
        category_accuracy = _compute_category_accuracy(y_test_true, y_test_pred)
        coverage_95 = _compute_coverage_95(y_test_true, y_test_lower, y_test_upper)
        boundary_error_rate = _compute_boundary_error_rate(y_test_true, y_test_pred)
        gap_metrics = {
            "gap_mae": test_metrics["mae"] - train_metrics["mae"],
            "gap_rmse": test_metrics["rmse"] - train_metrics["rmse"],
            "gap_mape": test_metrics["mape"] - train_metrics["mape"],
        }

        mlflow.log_metrics({
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "mape": test_metrics["mape"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_mape": test_metrics["mape"],
            "category_accuracy": category_accuracy,
            "coverage_95": coverage_95,
            "boundary_error_rate": boundary_error_rate,
            "test_category_accuracy": category_accuracy,
            "test_coverage_95": coverage_95,
            "test_boundary_error_rate": boundary_error_rate,
            **gap_metrics,
        })
        logger.info(
            f"Prophet {city}: test_MAE={test_metrics['mae']:.2f} "
            f"train_MAE={train_metrics['mae']:.2f} "
            f"gap_MAE={gap_metrics['gap_mae']:.2f} "
            f"CatAcc={category_accuracy:.1f}% "
            f"Coverage95={coverage_95:.1f}% "
            f"BoundaryErr={boundary_error_rate:.1f}%"
        )

        # Artifacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MODELS_DIR / "tmp"
        tmp.mkdir(exist_ok=True)

        plot_forecast(model, forecast, city,
                      str(tmp / "forecast.png"))
        mlflow.log_artifact(str(tmp / "forecast.png"))

        plot_residuals(y_test_true, y_test_pred, city,
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

        return model, {
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "mape": test_metrics["mape"],
            "category_accuracy": category_accuracy,
            "coverage_95": coverage_95,
            "boundary_error_rate": boundary_error_rate,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            **gap_metrics,
            "version": registered.version,
        }
    

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

    import argparse

    parser = argparse.ArgumentParser(
        description="Train AQI Prophet + SARIMA models"
    )
    parser.add_argument(
        "--city",
        type    = str,
        default = None,
        help = "City to train (default: all cities)",
    )
    args = parser.parse_args()




    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f"MLflow: {MLFLOW_URI}")
    print(f"MLflow version: {mlflow.__version__}\n")


    cities_to_train = [args.city] if args.city else list(CITIES.keys())
    print(f"Cities to train: {', '.join(cities_to_train)}")
    
    summary = {}

    for city in cities_to_train:
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
        if os.getenv("SKIP_SARIMA", "false").lower() != "true":
            print(f"  Training SARIMA...")
            sarima_m = train_sarima(city, df)
            print(f"  SARIMA  MAE: {sarima_m['mae']:.2f}")

        else:
            print(f"  Skipping SARIMA (set SKIP_SARIMA=false to enable)")
            sarima_m = {"mae": float(9999), "rmse": float(9999), "mape": float(9999)}

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
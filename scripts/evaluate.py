"""
scripts/evaluate.py

Evaluate champion Prophet models against held-out test data.
Computes MAE, RMSE, MAPE for each city and generates a report.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --city Delhi
    python scripts/evaluate.py --city Delhi --days 30
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import mlflow
import mlflow.prophet

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


from src.config import CITIES, get_aqi_category

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)s %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


#--------Config---------

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_DIR   = Path(os.getenv("DATA_DIR", "data"))
RESULTS_DIR = project_root / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#-------Metrics---------
def mean_absolute_error(y_true, y_pred):
    """MAE — average absolute error in AQI points."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred):
    """RMSE — penalises large errors more than MAE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE — percentage error.
    Clips y_true to avoid division by zero.
    """
    y_true_safe = np.clip(np.abs(y_true), 1, None)
    return float(np.mean(np.abs(y_true - y_pred) / y_true_safe) * 100)


def coverage_score(y_true, y_lower, y_upper):
    """
    Confidence interval coverage.
    What fraction of true values fall within [lower, upper].
    Target: ~95% for 95% confidence intervals.
    """
    covered = ((y_true >= y_lower) & (y_true <= y_upper)).mean()
    return float(covered * 100)


def category_accuracy(y_true, y_pred):
    """
    AQI category accuracy.
    Measures if predicted category matches true category.
    More interpretable than raw MAE for non-technical users.
    """
    true_cats = [get_aqi_category(int(v))["label"] for v in y_true]
    pred_cats = [get_aqi_category(int(v))["label"] for v in y_pred]
    correct   = sum(t == p for t, p in zip(true_cats, pred_cats))
    return float(correct / len(true_cats) * 100)


def boundary_cross_error_rate(y_true, y_pred, margin=10):
    """
    Category mismatches that occur near AQI category boundaries.
    Useful for explaining category accuracy dips caused by threshold effects.
    """
    boundaries = np.array([50, 100, 200, 300, 400], dtype=float)

    true_cats = np.array([get_aqi_category(int(v))["label"] for v in y_true])
    pred_cats = np.array([get_aqi_category(int(v))["label"] for v in y_pred])
    mismatches = true_cats != pred_cats

    if len(y_true) == 0:
        return 0.0

    dist_to_boundary = np.min(np.abs(y_true.reshape(-1, 1) - boundaries), axis=1)
    near_boundary = dist_to_boundary <= margin
    rate = np.mean(mismatches & near_boundary) * 100
    return float(rate)


def boundary_zone_mask(y_true, margin=10):
    """Mask of points whose actual AQI is within ±margin of a category boundary."""
    boundaries = np.array([50, 100, 200, 300, 400], dtype=float)
    if len(y_true) == 0:
        return np.array([], dtype=bool)
    dist_to_boundary = np.min(np.abs(y_true.reshape(-1, 1) - boundaries), axis=1)
    return dist_to_boundary <= margin


def interval_calibration_bins(y_true, y_lower, y_upper, n_bins=5):
    """
    Calibration by predicted interval width quantiles.
    Returns per-bin empirical coverage to diagnose over/under-confidence.
    """
    widths = y_upper - y_lower

    if len(widths) == 0:
        return []

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(widths, quantiles)
    bins = []

    for i in range(n_bins):
        low = float(edges[i])
        high = float(edges[i + 1])
        if i < n_bins - 1:
            mask = (widths >= low) & (widths < high)
        else:
            mask = (widths >= low) & (widths <= high)

        count = int(mask.sum())
        if count == 0:
            coverage = None
        else:
            coverage = float(((y_true[mask] >= y_lower[mask]) & (y_true[mask] <= y_upper[mask])).mean() * 100)

        bins.append({
            "bin": i + 1,
            "width_low": round(low, 2),
            "width_high": round(high, 2),
            "count": count,
            "coverage": round(coverage, 1) if coverage is not None else None,
        })

    return bins


#-----Data Loading--------

def load_test_data(city: str, test_days: int = 30) -> pd.DataFrame:
    """
    Load held-out test data for a city.
    Uses the last `test_days` days of processed features.

    Returns DataFrame with columns: ds, y
    """
    city_key  = city.lower().replace(" ", "_")
    feat_path = DATA_DIR / "processed" / city_key / "features.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features not found for {city}: {feat_path}. "
            "Run pipeline.py first."
        )

    df = pd.read_parquet(feat_path)

    # Use last test_days as held-out test set
    cutoff = df["ds"].max() - timedelta(days=test_days)
    test   = df[df["ds"] > cutoff].copy()

    if len(test) < 24:
        raise ValueError(
            f"Insufficient test data for {city}: "
            f"{len(test)} rows (need ≥ 24)"
        )

    logger.info(
        f"{city}: test set "
        f"{test['ds'].min().date()} → {test['ds'].max().date()} "
        f"({len(test)} rows)"
    )

    return test


def load_champion_model(city: str):
    """Load champion Prophet model from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client     = mlflow.MlflowClient()
    model_name = f"aqi_prophet_{city.lower().replace(' ', '_')}"

    try:
        alias_info = client.get_model_version_by_alias(
            model_name, "champion"
        )
        model_uri = (
            f"models:/{model_name}@champion"
        )
        model = mlflow.prophet.load_model(model_uri)
        logger.info(
            f"Loaded champion model for {city} "
            f"(version {alias_info.version})"
        )
        return model, alias_info.version

    except Exception as e:
        logger.error(f"Failed to load model for {city}: {e}")
        return None, None



#----Evaluation--------

def evaluate_city(city: str,
                  model,
                  test_df: pd.DataFrame) -> dict:
    """
    Evaluate Prophet model on held-out test data.

    Strategy — rolling forecast:
      For each test hour, use the previous hour's actual AQI
      as the lag regressor (simulates real inference conditions)
    """
    logger.info(f"Evaluating {city}...")

    # Build prediction dataframe with exactly the regressors expected by the model.
    # Any missing regressor is filled from test-set mean (or 0.0 fallback).
    model_regressors = list(model.extra_regressors.keys())
    future = test_df[["ds"]].copy()

    for reg in model_regressors:
        if reg in test_df.columns:
            fill_value = float(test_df[reg].mean()) if test_df[reg].notna().any() else 0.0
            future[reg] = test_df[reg].fillna(fill_value)
        else:
            future[reg] = 0.0

    # Generate predictions
    forecast = model.predict(future)
    forecast["yhat"]       = np.clip(forecast["yhat"],       0, 500)
    forecast["yhat_lower"] = np.clip(forecast["yhat_lower"], 0, 500)
    forecast["yhat_upper"] = np.clip(forecast["yhat_upper"], 0, 500)

    y_true  = test_df["y"].values
    y_pred  = forecast["yhat"].values
    y_lower = forecast["yhat_lower"].values
    y_upper = forecast["yhat_upper"].values

    # Compute all metrics
    mae       = mean_absolute_error(y_true, y_pred)
    rmse      = root_mean_squared_error(y_true, y_pred)
    mape      = mean_absolute_percentage_error(y_true, y_pred)
    coverage  = coverage_score(y_true, y_lower, y_upper)
    cat_acc   = category_accuracy(y_true, y_pred)
    boundary_err = boundary_cross_error_rate(y_true, y_pred, margin=10)
    calib_bins = interval_calibration_bins(y_true, y_lower, y_upper, n_bins=5)
    in_boundary_zone = boundary_zone_mask(y_true, margin=10)
    non_boundary_mask = ~in_boundary_zone

    if non_boundary_mask.any():
        cat_acc_ex_boundary = category_accuracy(
            y_true[non_boundary_mask], y_pred[non_boundary_mask]
        )
        cov_ex_boundary = coverage_score(
            y_true[non_boundary_mask],
            y_lower[non_boundary_mask],
            y_upper[non_boundary_mask],
        )
    else:
        cat_acc_ex_boundary = 0.0
        cov_ex_boundary = 0.0

    calibration_gap = max(0.0, 95.0 - coverage)
    category_error_rate = max(0.0, 100.0 - cat_acc)
    boundary_share = (
        boundary_err / category_error_rate
        if category_error_rate > 0 else 0.0
    )

    # Peak hour analysis (rush hours 7-10, 17-20)
    test_df_copy    = test_df.copy()
    test_df_copy["yhat"] = y_pred
    rush_mask       = test_df_copy["ds"].dt.hour.isin(
        list(range(7, 11)) + list(range(17, 21))
    )
    rush_mae = mean_absolute_error(
        test_df_copy.loc[rush_mask, "y"].values,
        test_df_copy.loc[rush_mask, "yhat"].values,
    ) if rush_mask.sum() > 0 else None

    result = {
        "city":              city,
        "test_rows":         len(test_df),
        "test_from":         str(test_df["ds"].min().date()),
        "test_to":           str(test_df["ds"].max().date()),
        "mae":               round(mae,      2),
        "rmse":              round(rmse,     2),
        "mape":              round(mape,     2),
        "coverage_95":       round(coverage, 1),
        "calibration_gap":   round(calibration_gap, 1),
        "category_accuracy": round(cat_acc,  1),
        "boundary_error_rate": round(boundary_err, 1),
        "boundary_share_of_errors": round(boundary_share, 3),
        "category_accuracy_ex_boundary": round(cat_acc_ex_boundary, 1),
        "coverage_95_ex_boundary": round(cov_ex_boundary, 1),
        "calibration_bins":  calib_bins,
        "rush_hour_mae":     round(rush_mae, 2) if rush_mae else None,
        "mean_actual_aqi":   round(float(y_true.mean()), 1),
        "mean_pred_aqi":     round(float(y_pred.mean()), 1),
    }

    logger.info(
        f"{city}: MAE={mae:.2f} | RMSE={rmse:.2f} | "
        f"MAPE={mape:.2f}% | Coverage={coverage:.1f}% | "
        f"CatAcc={cat_acc:.1f}% | BoundaryErr={boundary_err:.1f}% | "
        f"CalGap={calibration_gap:.1f}% | BndShare={boundary_share:.2f} | "
        f"CatAccExBnd={cat_acc_ex_boundary:.1f}% | CovExBnd={cov_ex_boundary:.1f}%"
    )

    return result



#---Report Generation---

def print_report(results: list[dict]):
    """Print formatted evaluation report to console."""
    print()
    print("=" * 75)
    print("AQI PREDICTION SYSTEM — MODEL EVALUATION REPORT")
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 75)
    print()
    print(f"{'City':<12} {'MAE':>6} {'RMSE':>7} {'MAPE':>7} "
          f"{'Coverage':>10} {'Cat Acc':>9} {'Bnd Err':>9} {'Rush MAE':>10}")
    print("-" * 75)

    for r in results:
        rush = f"{r['rush_hour_mae']:.2f}" if r["rush_hour_mae"] else "N/A"
        print(
            f"{r['city']:<12} "
            f"{r['mae']:>6.2f} "
            f"{r['rmse']:>7.2f} "
            f"{r['mape']:>6.2f}% "
            f"{r['coverage_95']:>9.1f}% "
            f"{r['category_accuracy']:>8.1f}% "
            f"{r['boundary_error_rate']:>8.1f}% "
            f"{rush:>10}"
        )

    print("-" * 75)

    # Aggregates
    maes  = [r["mae"]  for r in results]
    rmses = [r["rmse"] for r in results]
    mapes = [r["mape"] for r in results]
    covs  = [r["coverage_95"] for r in results]

    print(
        f"{'Average':<12} "
        f"{np.mean(maes):>6.2f} "
        f"{np.mean(rmses):>7.2f} "
        f"{np.mean(mapes):>6.2f}% "
        f"{np.mean(covs):>9.1f}% "
    )
    print("=" * 75)
    print()
    print("Metric definitions:")
    print("  MAE      — Mean Absolute Error (AQI points)")
    print("  RMSE     — Root Mean Squared Error (penalises large errors)")
    print("  MAPE     — Mean Absolute Percentage Error (%)")
    print("  Coverage — % of true values within 95% confidence interval")
    print("  Cat Acc  — % of hours where predicted AQI category matches actual")
    print("  Bnd Err  — % category errors where actual AQI is near a threshold")
    print("  Rush MAE — MAE during morning/evening rush hours (7-10, 17-20)")
    print()

    print("Calibration by interval-width bins (target coverage ~95%):")
    for r in results:
        bins = r.get("calibration_bins", [])
        bins_str = ", ".join([
            f"B{b['bin']}={b['coverage']}%"
            for b in bins if b.get("coverage") is not None
        ])
        print(f"  {r['city']:<12} {bins_str}")
    print()

    print("Compact summary scores:")
    print("  calibration_gap = 95 - coverage")
    print("  boundary_share_of_errors = boundary_error_rate / (100 - category_accuracy)")
    for r in results:
        print(
            f"  {r['city']:<12} "
            f"calibration_gap={r['calibration_gap']:.1f}% | "
            f"boundary_share_of_errors={r['boundary_share_of_errors']:.2f} | "
            f"category_accuracy_ex_boundary={r['category_accuracy_ex_boundary']:.1f}% | "
            f"coverage_95_ex_boundary={r['coverage_95_ex_boundary']:.1f}%"
        )
    print()

    # Acceptance criteria
    print("Acceptance Criteria:")
    all_passed = True
    for r in results:
        passed = r["mae"] <= 50
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        print(f"  {r['city']:<12} MAE ≤ 50: {r['mae']:>6.2f} {status}")

    print()
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 75)


def save_report(results: list[dict]):
    """Save evaluation results to JSON and CSV."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = RESULTS_DIR / f"evaluation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "mlflow_uri":   MLFLOW_URI,
            "results":      results,
        }, f, indent=2)

    # CSV report
    csv_path = RESULTS_DIR / f"evaluation_{timestamp}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    # Latest symlink — always points to most recent report
    latest_json = RESULTS_DIR / "evaluation_latest.json"
    latest_csv  = RESULTS_DIR / "evaluation_latest.csv"

    with open(latest_json, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "mlflow_uri":   MLFLOW_URI,
            "results":      results,
        }, f, indent=2)

    pd.DataFrame(results).to_csv(latest_csv, index=False)

    logger.info(f"Report saved: {json_path}")
    logger.info(f"Report saved: {csv_path}")
    logger.info(f"Latest:       {latest_json}")



#----Main-----------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AQI champion models"
    )
    parser.add_argument(
        "--city",
        type    = str,
        default = None,
        help    = "City to evaluate (default: all cities)"
    )
    parser.add_argument(
        "--days",
        type    = int,
        default = 30,
        help    = "Number of test days (default: 30)"
    )
    parser.add_argument(
        "--no-save",
        action  = "store_true",
        help    = "Do not save report to disk"
    )
    args = parser.parse_args()

    cities = [args.city] if args.city else list(CITIES.keys())

    logger.info(f"Evaluating cities: {cities}")
    logger.info(f"Test window: last {args.days} days")
    logger.info(f"MLflow URI: {MLFLOW_URI}")

    results = []
    failed  = []

    for city in cities:
        try:
            # Load model
            model, version = load_champion_model(city)
            if model is None:
                logger.error(f"Skipping {city} — model not loaded")
                failed.append(city)
                continue

            # Load test data
            test_df = load_test_data(city, test_days=args.days)

            # Evaluate
            result          = evaluate_city(city, model, test_df)
            result["version"] = version
            results.append(result)

        except FileNotFoundError as e:
            logger.error(f"Skipping {city}: {e}")
            failed.append(city)
        except Exception as e:
            logger.error(f"Evaluation failed for {city}: {e}",
                         exc_info=True)
            failed.append(city)

    if not results:
        logger.error("No cities evaluated successfully")
        sys.exit(1)

    # Print report
    print_report(results)

    # Save report
    if not args.no_save:
        save_report(results)

    if failed:
        logger.warning(f"Failed cities: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
KS-test drift detection.
Compares last 7 days of live AQI against the training baseline.
Called by:
  - Airflow hourly DAG  -> check after each ingestion
  - FastAPI /drift -> expose drift status via API
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from scipy.stats import ks_2samp
from dotenv import load_dotenv

load_dotenv()


try:
    from ..config import CITIES
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES


logger   = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Drift is declared when KS p-value falls below this threshold
# 0.05 = 5% significance level — standard statistical convention
DRIFT_THRESHOLD = 0.05


# Minimum number of recent samples needed for a valid KS-test
# Below this we cannot draw reliable conclusions
MIN_SAMPLES = 24   # at least 1 day of hourly readings


def get_baseline(city):
    """
    Load the saved baseline AQI distribution for a city.
    This was saved by pipeline.py during initial setup.
    Returns Series of AQI values or None if baseline not found.
    """

    city_key = city.lower().replace(" ", "_")
    baseline_path = DATA_DIR / "baseline" / f"{city_key}_baseline.parquet"

    if not baseline_path.exists():
        logger.warning(
            f"Baseline not found for {city} at {baseline_path}. "
            "Run pipeline.py first."
        )

        return None
    
    df = pd.read_parquet(baseline_path)

    return df["aqi"].dropna()

def get_recent_window(city, days=7):
    """
    Load the last N days of raw AQI data for a city.
    Reads directly from data/raw/{city}/date=*.parquet files.
    Returns Series of AQI values or None if insufficient data.
    """

    city_key = city.lower().replace(" ","_")
    raw_dir  = DATA_DIR / "raw" / city_key
    cutoff   = datetime.now(timezone.utc) - timedelta(days=days)

    parquet_files = sorted(raw_dir.glob("date=*.parquet"), reverse=True)

    if not parquet_files:
        logger.warning(f"No raw data found for {city}")
        return None
    
    dfs = []

    for f in parquet_files:

        try:
            df = pd.read_parquet(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            recent = df[df["timestamp"] >= cutoff]

            if not recent.empty:
                dfs.append(recent)

        except Exception as e:
            logger.warning(f"Could not read {f.name}: {e}")
            continue

    
    if not dfs:
        logger.warning(f"No recent data within {days} days for {city}")
        return None
    
    combined = (pd.concat(dfs)
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                )
    
    return combined["aqi"].dropna()


def save_drift_window(city, recent):
    """
    Save recent AQI window to data/drift/ for API access.
    FastAPI reads this file to show drift status on the dashboard.
    """
    
    drift_dir = DATA_DIR / "drift"
    drift_dir.mkdir(parents=True,exist_ok=True)
    city_key = city.lower().replace(" ","_")

    df = pd.DataFrame({"aqi": recent.values,
                       "checked_at": datetime.now(timezone.utc).isoformat()})
    
    df.to_parquet(drift_dir / f"{city_key}_recent.parquet", index=False)


def check_drift(city, window_days=7):
    """
    Run KS-test for one city.

    Algorithm:
      1. Load baseline AQI distribution (from training data)
      2. Load last window_days of live AQI readings
      3. Run two-sample KS-test
      4. If p_value < 0.05 -> distributions differ -> drift detected

    Returns dict with:
      city         — city name
      p_value      — KS-test p-value (lower = more drift)
      ks_stat      — KS statistic (higher = more drift)
      is_drifted   — bool, True if p_value < DRIFT_THRESHOLD
      baseline_mean — mean AQI in training data
      recent_mean   — mean AQI in recent window
      recent_n      — number of recent samples used
      checked_at   — UTC timestamp of this check
      error        — error message if check failed, else None
    """

    result = {
        "city":           city,
        "p_value":        1.0,
        "ks_stat":        0.0,
        "is_drifted":     False,
        "baseline_mean":  None,
        "recent_mean":    None,
        "recent_n":       0,
        "checked_at":     datetime.now(timezone.utc).isoformat(),
        "error":          None,
    }


    try:
        #Load baseline

        baseline = get_baseline(city)

        if baseline is None or baseline.empty:
            result["error"] = "Baseline not found — run pipeline.py first"
            return result

        # Load recent window
        recent = get_recent_window(city, days=window_days)

        if recent is None or len(recent) < MIN_SAMPLES:
            result["error"] = (
                f"Insufficient recent data: "
                f"{len(recent) if recent is not None else 0} samples "
                f"(need {MIN_SAMPLES})"
            )
            return result
        
        # Save for API access
        save_drift_window(city, recent)

        # Run KS-test
        # ks_2samp returns (statistic, p_value)
        # statistic = max difference between CDFs (0 to 1)
        # p_value   = probability of seeing this difference by chance
        # low p_value = distributions are genuinely different
        
        ks_stat, p_value = ks_2samp(baseline.values, recent.values)

        result["p_value"] = round(float(p_value),  6)
        result["ks_stat"] = round(float(ks_stat),  4)
        result["is_drifted"] = bool(p_value < DRIFT_THRESHOLD)
        result["baseline_mean"] = round(float(baseline.mean()), 1)
        result["recent_mean"] = round(float(recent.mean()),   1)
        result["recent_n"] = int(len(recent))


        if result["is_drifted"]:
            logger.warning(
                f"DRIFT DETECTED — {city}: "
                f"p={p_value:.4f}, KS={ks_stat:.4f} | "
                f"baseline_mean={result['baseline_mean']} "
                f"recent_mean={result['recent_mean']}"
            )

        else:
            logger.info(
                f"No drift — {city}: "
                f"p={p_value:.4f}, KS={ks_stat:.4f} | "
                f"recent_mean={result['recent_mean']}"
            )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Drift check failed for {city}: {e}", exc_info=True)


    return result

def check_all_cities():
    """
    Run drift check for all 5 cities.
    Returns dict of {city: result_dict}.
    Called by Airflow hourly DAG after ingestion.
    """

    results = {}

    for city in CITIES:
        results[city] = check_drift(city)
    
    return results

def should_retrain(results):
    """
    Given drift results for all cities, return list of cities
    that need retraining.
    Called by Airflow DAG branch operator.
    """

    return [city for city, result in results.items() 
            if result.get("is_drifted") and not result.get("error")]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    print("Running drift detection for all cities...\n")
    results = check_all_cities()

    print(f"{'City':<12} {'p-value':<10} {'KS stat':<10} "
          f"{'Status':<10} {'Baseline':<10} {'Recent':<10} {'N'}")
    print("-" * 72)

    for city, r in results.items():
        if r.get("error"):
            print(f"{city:<12} ERROR: {r['error']}")
            continue

        status = "DRIFT ⚠" if r["is_drifted"] else "OK ✓"
        print(
            f"{city:<12} "
            f"{r['p_value']:<10.4f} "
            f"{r['ks_stat']:<10.4f} "
            f"{status:<10} "
            f"{r['baseline_mean']:<10} "
            f"{r['recent_mean']:<10} "
            f"{r['recent_n']}"
        )

    drifted = should_retrain(results)
    if drifted:
        print(f"\nCities requiring retraining: {drifted}")
    else:
        print(f"\nNo retraining required.")
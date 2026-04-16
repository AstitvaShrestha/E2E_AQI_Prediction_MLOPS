"""
Spark feature engineering pipeline.
Reads raw AQI + weather Parquet files, computes lag features,
rolling statistics, and cyclical calendar encodings.
Outputs processed features per city ready for Prophet training.
"""


import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from ..config import CITIES


logger = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

def build_features_spark(city):
    """
    Build feature set using PySpark.
    Falls back to pandas if Spark unavailable (e.g., unit tests).
    """

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        spark = (
            SparkSession.builder
            .appName(f"AQIPrediction-Features-{city}")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")
        return _build_with_spark(spark, city)
    
    except ImportError:
        logger.warning("PySpark not available — falling back to pandas")
        return _build_with_pandas(city)
    

def _build_with_spark(spark, city):
    """Spark implementation of feature engineering pipeline."""
    
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    city_key = city.lower().replace(" ", "_")
    raw_path = DATA_DIR / "raw" / city_key / "*.parquet"

    # Load AQI data

    aqi_df = spark.read.parquet(str(raw_path)).select(
        F.col("timestamp").cast("timestamp"),
        F.col("aqi").cast("integer"),
        F.col("city").cast("string")
    )


    # Load weather data
    weather_path = DATA_DIR / "raw" / city_key / "weather" / "*.parquet"

    try:
        weather_df = spark.read.parquet(str(weather_path)).select(
            F.col("timestamp").cast("timestamp"),
            F.col("temperature_2m").cast("double"),
            F.col("relative_humidity_2m").cast("double"),
            F.col("wind_speed_10m").cast("double"),
            F.col("surface_pressure").cast("double")
        )

        df = aqi_df.join(weather_df, on="timestamp", how="left")
    
    except Exception as e:
        logger.warning(f"No weather data found for {city}, using AQI only")
        df = aqi_df
        df = df.withColumn("temperature_2m", F.lit(None).cast("double"))
        df = df.withColumn("wind_speed_10m", F.lit(None).cast("double"))
        df = df.withColumn("relative_humidity_2m", F.lit(None).cast("double"))
        df = df.withColumn("surface_pressure", F.lit(None).cast("double"))

    
    # Sort by time
    df = df.orderBy("timestamp")

    # Window spec ordered by timestamp
    w = Window.orderBy("timestamp")

    # Lag features
    for lag_h in [1, 2, 24, 48, 168]:
        df = df.withColumn(f"aqi_lag_{lag_h}h", F.lag("aqi", lag_h).over(w))
    
    # Rolling window statistics (using rangebetween on row count as proxy)
    for hours, name in [(6, "6h"), (24, "24h"), (168, "7d")]:
        w_roll = Window.orderBy("timestamp").rowsBetween(-hours, -1)
        df = df.withColumn(f"aqi_roll_mean_{name}", F.avg("aqi").over(w_roll))
        df = df.withColumn(f"aqi_roll_std_{name}",  F.stddev("aqi").over(w_roll))
    
    df = df.withColumn("aqi_roll_max_24h", 
                       F.max("aqi").over(
                           Window.orderBy("timestamp").rowsBetween(-24, -1)
                       )
                    )
    
    # Calendar features
    df = df.withColumn("hour",      F.hour("timestamp"))
    df = df.withColumn("dayofweek", F.dayofweek("timestamp"))
    df = df.withColumn("month",     F.month("timestamp"))
    df = df.withColumn("is_weekend",
                       (F.dayofweek("timestamp").isin([1, 7])).cast("integer"))
    df = df.withColumn("is_rush_hour",
                       (F.hour("timestamp").between(7, 10) |
                        F.hour("timestamp").between(17, 20)).cast("integer"))

    # Cyclical encodings
    pi = float(np.pi)
    df = df.withColumn("hour_sin",  F.sin(2 * pi * F.col("hour") / 24))
    df = df.withColumn("hour_cos",  F.cos(2 * pi * F.col("hour") / 24))
    df = df.withColumn("dow_sin",   F.sin(2 * pi * F.col("dayofweek") / 7))
    df = df.withColumn("dow_cos",   F.cos(2 * pi * F.col("dayofweek") / 7))
    df = df.withColumn("month_sin", F.sin(2 * pi * F.col("month") / 12))
    df = df.withColumn("month_cos", F.cos(2 * pi * F.col("month") / 12))


    # Rename for Prophet
    df = df.withColumnRenamed("timestamp", "ds").withColumnRenamed("aqi", "y")


    # Drop rwos with null target
    df = df.dropna(subset=["y"])

    # Convert to pandas for MLflow / Prophet
    pdf = df.toPandas()
    pdf["ds"] = pd.to_datetime(pdf["ds"])


    # Fill weather NaNs with city mean (fallback)
    for col in ["temperature_2m", "relative_humidity_2m", 
                "wind_speed_10m", "surface_pressure"]:
        if col in pdf.columns:
            mean_val = pdf[col].mean()
            pdf[col] = pdf[col].fillna(mean_val)

    return pdf.sort_values("ds").reset_index(drop=True)


def _build_with_pandas(city):
    """Pandas implementation of feature engineering pipeline."""

    city_key = city.lower().replace(" ", "_")
    raw_dir  = DATA_DIR / "raw" / city_key

    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No raw data found for {city} at {raw_dir}")

    aqi_df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"])
    aqi_df = (aqi_df.sort_values("timestamp")
              .drop_duplicates(subset=["timestamp"])
              .reset_index(drop=True))

    # Weather
    weather_files = list((raw_dir / "weather").glob("*.parquet")) \
        if (raw_dir / "weather").exists() else []
    
    if weather_files:
        weather_df = pd.concat([pd.read_parquet(f) for f in weather_files])
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        weather_df = weather_df.drop_duplicates(subset=["timestamp"])
        df = aqi_df.merge(weather_df[
            ["timestamp", "temperature_2m", "wind_speed_10m",
             "relative_humidity_2m", "precipitation"]
        ], on="timestamp", how="left")
    else:
        df = aqi_df.copy()
        for col in ["temperature_2m", "wind_speed_10m",
                    "relative_humidity_2m", "precipitation"]:
            df[col] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lag features
    for lag_h in [1, 2, 24, 48, 168]:
        df[f"aqi_lag_{lag_h}h"] = df["aqi"].shift(lag_h)

    # Rolling features
    for hours, name in [(6, "6h"), (24, "24h"), (168, "7d")]:
        df[f"aqi_roll_mean_{name}"] = df["aqi"].rolling(hours, min_periods=1).mean()
        df[f"aqi_roll_std_{name}"]  = df["aqi"].rolling(hours, min_periods=1).std()
    df["aqi_roll_max_24h"] = df["aqi"].rolling(24, min_periods=1).max()

    # Calendar
    df["hour"]       = df["timestamp"].dt.hour
    df["dayofweek"]  = df["timestamp"].dt.dayofweek
    df["month"]      = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = (
        df["hour"].between(7, 10) | df["hour"].between(17, 20)
    ).astype(int)

    # Cyclical
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)

    # Prophet column names
    df = df.rename(columns={"timestamp": "ds", "aqi": "y"})
    df["city"] = city

    # Fill weather NaNs
    for col in ["temperature_2m", "wind_speed_10m",
                "relative_humidity_2m", "precipitation"]:
        df[col] = df[col].fillna(df[col].mean())

    return df.dropna(subset=["y"]).reset_index(drop=True)


def save_features(city, df):
    """Save processed features to Parquet — DVC tracks this file."""

    out_dir = DATA_DIR / "processed" / city.lower().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved features for {city}: {len(df)} rows -> {out_path}")
    return out_path


def compute_baseline_stats(city, df):
    """
    Compute and save baseline distribution statistics.
    Called once during initial setup — used for drift detection.
    """

    baseline_dir = DATA_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "city":           city,
        "computed_at":    datetime.utcnow().isoformat(),
        "aqi_mean":       float(df["y"].mean()),
        "aqi_std":        float(df["y"].std()),
        "aqi_median":     float(df["y"].median()),
        "aqi_p25":        float(df["y"].quantile(0.25)),
        "aqi_p75":        float(df["y"].quantile(0.75)),
        "wind_mean":      float(df["wind_speed_10m"].mean())
                          if "wind_speed_10m" in df.columns else 5.0,
        "temp_mean":      float(df["temperature_2m"].mean())
                          if "temperature_2m" in df.columns else 25.0,
        "n_samples":      int(len(df)),
    }


    # Save raw baseline AQI distribution for KS-test
    baseline_df = df[["ds", "y"]].rename(columns={"y": "aqi"}).copy()
    baseline_df.to_parquet(
        baseline_dir / f"{city.lower().replace(' ', '_')}_baseline.parquet",
        index=False,
    )

    with open(baseline_dir / f"{city.lower().replace(' ', '_')}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Baseline stats saved for {city}")
    return stats

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for city in CITIES:
        try:
            df = build_features_spark(city)
            save_features(city, df)
            compute_baseline_stats(city, df)
        except Exception as e:
            logger.error(f"Feature pipeline failed for {city}: {e}")
"""
Spark feature engineering pipeline.
Reads raw AQI + weather Parquet files, computes lag features,
rolling statistics, and cyclical calendar encodings.
Outputs processed features per city ready for Prophet training.
"""


import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

try:
    from ..config import CITIES
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES


logger = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))


# Weather columns we expect from Kaggle CSV or OpenMeteo API
WEATHER_COLS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
]

# Bonus India-specific columns from Kaggle dataset
# Present only in Kaggle-seeded data — handled gracefully if missing
BONUS_COLS = [
    "temp_inversion",
    "crop_burning",
    "festival_period",
]

# Lag hours — how far back Prophet looks for autocorrelation signal
LAG_HOURS = [1, 2, 3, 24, 48, 168]   # 168 = 1 week



def build_features_spark(city: str) -> pd.DataFrame:
    """
    Build feature set using PySpark.
    Falls back to pandas automatically if Spark/Java unavailable.
    """
    try:
        import pyspark
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder
            .appName(f"AirCast-{city}")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        logger.info(f"Using Spark {spark.version} for {city}")
        df = _build_with_spark(spark, city)
        spark.stop()
        return df
    except Exception as e:
        logger.warning(f"Spark unavailable ({e}) — using pandas for {city}")
        return _build_with_pandas(city)

def _reindex_to_hourly(df: pd.DataFrame,
                       timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Reindex DataFrame to complete hourly grid.
    Ensures every hour exists between first and last date.
    Rounds to day boundaries so 00:00 and 23:00 are always included.
    Missing hours get NaN for AQI/PM2.5 — Prophet handles these.
    City column is forward-filled (safe scalar).
    """
    df = df.set_index(timestamp_col)

    # Round to day boundaries — ensures 00:00 and 23:00 included
    start = df.index.min().floor("D")
    end   = df.index.max().ceil("D") - pd.Timedelta(hours=1)

    full_range = pd.date_range(
        start=start,
        end=end,
        freq="h",
        tz="UTC"
    )

    df = df.reindex(full_range)
    df.index.name = timestamp_col
    df = df.reset_index()

    # Forward fill city — scalar string column, safe to fill
    if "city" in df.columns:
        df["city"] = df["city"].ffill()

    return df

def _build_with_spark(spark, city: str) -> pd.DataFrame:
  
  
    city_key = city.lower().replace(" ", "_")
    raw_path = str(DATA_DIR / "raw" / city_key / "date=*.parquet")

    try:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        aqi_df = spark.read.parquet(raw_path)
        
    except Exception as e:
        raise FileNotFoundError(
            f"No raw AQI data for {city} at {raw_path}."
        ) from e

    aqi_df = (aqi_df
              .select(
                  # Floor to hour — fixes Kaggle(HH:00) vs OpenAQ(HH:30)
                  F.date_trunc("hour",
                      F.col("timestamp").cast("timestamp")
                  ).alias("timestamp"),
                  F.col("aqi").cast("double").alias("aqi"),
                  F.col("pm25").cast("double").alias("pm25"),
              )
              .dropna(subset=["timestamp", "aqi"])
              # Deduplicate after floor — same hour from multiple sources
              .dropDuplicates(["timestamp"])
              .orderBy("timestamp"))

    # Weather
    weather_path = str(
        DATA_DIR / "raw" / city_key / "weather" / "date=*.parquet"
    )
    try:
        weather_df = spark.read.parquet(weather_path)

        # Rename Kaggle columns
        col_map = {
            "Temp_2m_C":          "temperature_2m",
            "Wind_Speed_10m_kmh": "wind_speed_10m",
            "Humidity_Percent":   "relative_humidity_2m",
            "Precipitation_mm":   "precipitation",
            "Pressure_MSL_hPa":   "surface_pressure",
        }
        for old, new in col_map.items():
            if old in weather_df.columns:
                weather_df = weather_df.withColumnRenamed(old, new)

        # Wind speed unit fix
        if "wind_speed_10m" in weather_df.columns:
            sample_wind = weather_df.select(
                F.avg("wind_speed_10m")
            ).collect()[0][0]
            if sample_wind and sample_wind > 15:
                weather_df = weather_df.withColumn(
                    "wind_speed_10m",
                    F.col("wind_speed_10m") / 3.6
                )

        weather_select = ["timestamp"] + [
            c for c in WEATHER_COLS if c in weather_df.columns
        ]

        weather_df = (weather_df
                      .select(*[
                          F.date_trunc("hour",
                              F.col(c).cast("timestamp")
                          ).alias(c)
                          if c == "timestamp"
                          else F.col(c).cast("double")
                          for c in weather_select
                      ])
                      .dropna(subset=["timestamp"])
                      # Deduplicate after floor
                      .dropDuplicates(["timestamp"]))

        df = aqi_df.join(weather_df, on="timestamp", how="left")

    except Exception as e:
        logger.warning(f"No weather data for {city}: {e}")
        df = aqi_df
        for col in WEATHER_COLS:
            df = df.withColumn(col, F.lit(None).cast("double"))

    # Window spec
    w = Window.orderBy("timestamp")

    # Lag features
    for lag_h in LAG_HOURS:
        df = df.withColumn(
            f"aqi_lag_{lag_h}h",
            F.lag("aqi", lag_h).over(w)
        )

    # Rolling window statistics
    for hours, name in [(6, "6h"), (24, "24h"), (168, "7d")]:
        w_roll = Window.orderBy("timestamp").rowsBetween(-hours, -1)
        df = (df
              .withColumn(f"aqi_roll_mean_{name}", F.avg("aqi").over(w_roll))
              .withColumn(f"aqi_roll_std_{name}",  F.stddev("aqi").over(w_roll))
              .withColumn(f"aqi_roll_max_{name}",  F.max("aqi").over(w_roll)))

    # Calendar features
    pi = float(np.pi)
    df = (df
          .withColumn("hour",      F.hour("timestamp"))
          .withColumn("dayofweek", F.dayofweek("timestamp"))
          .withColumn("month",     F.month("timestamp"))
          .withColumn("is_weekend",
                      F.dayofweek("timestamp").isin([1,7]).cast("integer"))
          .withColumn("is_rush_hour",
                      (F.hour("timestamp").between(7,10) |
                       F.hour("timestamp").between(17,20)).cast("integer"))
          .withColumn("hour_sin",  F.sin(2*pi*F.col("hour")/24))
          .withColumn("hour_cos",  F.cos(2*pi*F.col("hour")/24))
          .withColumn("dow_sin",   F.sin(2*pi*F.col("dayofweek")/7))
          .withColumn("dow_cos",   F.cos(2*pi*F.col("dayofweek")/7))
          .withColumn("month_sin", F.sin(2*pi*F.col("month")/12))
          .withColumn("month_cos", F.cos(2*pi*F.col("month")/12)))

    # Prophet convention
    df = (df.withColumnRenamed("timestamp", "ds")
            .withColumnRenamed("aqi", "y")
            .dropna(subset=["y"]))

    # Convert to pandas
    pdf = df.toPandas()
    pdf["ds"]   = pd.to_datetime(pdf["ds"])
    pdf["city"] = city

    # Reindex to complete hourly grid in pandas
    # (Spark rowsBetween uses positions not clock-hours)
    pdf = pdf.rename(columns={"ds": "timestamp", "y": "aqi"})
    pdf = _reindex_to_hourly(pdf, timestamp_col="timestamp")
    pdf = pdf.rename(columns={"timestamp": "ds", "aqi": "y"})

    # Recompute lags on complete grid
    for lag_h in LAG_HOURS:
        pdf[f"aqi_lag_{lag_h}h"] = pdf["y"].shift(lag_h)

    for hours, name in [(6, "6h"), (24, "24h"), (168, "7d")]:
        pdf[f"aqi_roll_mean_{name}"] = pdf["y"].rolling(hours, min_periods=1).mean()
        pdf[f"aqi_roll_std_{name}"]  = pdf["y"].rolling(hours, min_periods=1).std()
        pdf[f"aqi_roll_max_{name}"]  = pdf["y"].rolling(hours, min_periods=1).max()

    # Fill weather NaN with city mean
    for col in WEATHER_COLS:
        if col in pdf.columns:
            pdf[col] = pdf[col].fillna(pdf[col].mean())

    return pdf.sort_values("ds").reset_index(drop=True)


# Pandas fallback
def _build_with_pandas(city: str) -> pd.DataFrame:
    """
    Pure pandas feature engineering — identical logic to Spark version.
    Used when Spark/Java is unavailable (CI, unit tests, low memory).
    """
    city_key = city.lower().replace(" ", "_")
    raw_dir  = DATA_DIR / "raw" / city_key

    # Load AQI
    aqi_files = sorted(raw_dir.glob("date=*.parquet"))
    if not aqi_files:
        raise FileNotFoundError(
            f"No AQI data found for {city} in {raw_dir}. "
            "Run seed_from_kaggle.py first."
        )

    aqi_df = pd.concat(
        [pd.read_parquet(f) for f in aqi_files],
        ignore_index=True
    )
    aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"], utc=True).dt.floor("h")
    aqi_df = (aqi_df
              .dropna(subset=["timestamp", "aqi"])
              .sort_values("timestamp")
              .drop_duplicates(subset=["timestamp"])
              .reset_index(drop=True))

    aqi_df = _reindex_to_hourly(aqi_df, timestamp_col="timestamp")

    logger.info(
        f"{city}: {len(aqi_df)} rows after reindex "
        f"({aqi_df['aqi'].isna().sum()} missing hours filled with NaN)"
    )

    # Load weather data
    weather_dir   = raw_dir / "weather"
    weather_files = sorted(weather_dir.glob("date=*.parquet")) \
                    if weather_dir.exists() else []

    if weather_files:
        weather_df = pd.concat(
            [pd.read_parquet(f) for f in weather_files],
            ignore_index=True
        )
        weather_df["timestamp"] = pd.to_datetime(
            weather_df["timestamp"], utc=True
        ).dt.floor("h")

        # Rename Kaggle column names if present
        col_map = {
            "Temp_2m_C":          "temperature_2m",
            "Wind_Speed_10m_kmh": "wind_speed_10m",
            "Humidity_Percent":   "relative_humidity_2m",
            "Precipitation_mm":   "precipitation",
            "Pressure_MSL_hPa":   "surface_pressure",
        }
        weather_df = weather_df.rename(columns=col_map)

        # Wind speed unit fix
        if "wind_speed_10m" in weather_df.columns:
            mean_wind = weather_df["wind_speed_10m"].mean()
            if mean_wind > 15:   # km/h values are typically > 15
                weather_df["wind_speed_10m"] = weather_df["wind_speed_10m"] / 3.6

        weather_df = (weather_df
                      .drop_duplicates(subset=["timestamp"])
                      .sort_values("timestamp"))

        # Merge AQI + weather on timestamp
        available_wx_cols = ["timestamp"] + [
            c for c in WEATHER_COLS + BONUS_COLS
            if c in weather_df.columns
        ]
        df = aqi_df.merge(
            weather_df[available_wx_cols],
            on="timestamp",
            how="left"
        )
    else:
        logger.warning(f"No weather data for {city} — using AQI only")
        df = aqi_df.copy()
        for col in WEATHER_COLS:
            df[col] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lag features
    for lag_h in LAG_HOURS:
        df[f"aqi_lag_{lag_h}h"] = df["aqi"].shift(lag_h)

    # Rolling window statistics
    for hours, name in [(6, "6h"), (24, "24h"), (168, "7d")]:
        df[f"aqi_roll_mean_{name}"] = (
            df["aqi"].rolling(hours, min_periods=1).mean()
        )
        df[f"aqi_roll_std_{name}"] = (
            df["aqi"].rolling(hours, min_periods=1).std()
        )
        df[f"aqi_roll_max_{name}"] = (
            df["aqi"].rolling(hours, min_periods=1).max()
        )

    # Calendar features
    df["hour"]         = df["timestamp"].dt.hour
    df["dayofweek"]    = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = (
        df["hour"].between(7, 10) |
        df["hour"].between(17, 20)
    ).astype(int)

    # Cyclical encodings
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)

    # Rename to Prophet convention
    df = df.rename(columns={"timestamp": "ds", "aqi": "y"})
    df["city"] = city

    # Fill weather NaN with city mean
    for col in WEATHER_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    return (df.dropna(subset=["y"])
              .sort_values("ds")
              .reset_index(drop=True))


def save_features(city, df):
    """
    Save processed feature DataFrame to Parquet - DVC tracks this file
    """
    out_dir = DATA_DIR / "processed" / city.lower().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(
        f"Saved features for {city}: {len(df)} rows → {out_path}"
    )
    return out_path


# Baseline stats for drift detection

def compute_baseline_stats(city, df):
    """
    Compute and save baseline distribution statistics.
    Called once during initial setup.
    Used by drift.py (KS-test) to detect when live data
    diverges from training distribution.
    """
    baseline_dir = DATA_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    city_key = city.lower().replace(" ", "_")

    stats = {
        "city":        city,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_samples":   int(len(df)),
        "aqi_mean":    round(float(df["y"].mean()),   2),
        "aqi_std":     round(float(df["y"].std()),    2),
        "aqi_median":  round(float(df["y"].median()), 2),
        "aqi_p25":     round(float(df["y"].quantile(0.25)), 2),
        "aqi_p75":     round(float(df["y"].quantile(0.75)), 2),
        "aqi_p95":     round(float(df["y"].quantile(0.95)), 2),
        "wind_mean":   round(float(df["wind_speed_10m"].mean()), 2)
                       if "wind_speed_10m" in df.columns else None,
        "temp_mean":   round(float(df["temperature_2m"].mean()), 2)
                       if "temperature_2m" in df.columns else None,
    }

    # Save raw AQI distribution for KS-test comparisons
    baseline_df = df[["ds", "y"]].rename(columns={"y": "aqi"}).copy()
    baseline_df.to_parquet(
        baseline_dir / f"{city_key}_baseline.parquet",
        index=False
    )

    with open(baseline_dir / f"{city_key}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Baseline saved for {city}: "
        f"mean={stats['aqi_mean']}, std={stats['aqi_std']}, "
        f"n={stats['n_samples']}"
    )
    return stats



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    for city in CITIES:
        print(f"\n{'='*50}")
        print(f"Processing {city}...")
        print(f"{'='*50}")

        try:
            # Build features
            df = build_features_spark(city)

            print(f"  Rows:     {len(df):,}")
            print(f"  Columns:  {len(df.columns)}")
            print(f"  Date range: {df['ds'].min().date()} "
                  f"-> {df['ds'].max().date()}")
            print(f"  AQI mean: {df['y'].mean():.1f}")
            print(f"  AQI max:  {df['y'].max():.1f}")

            # Verify key columns exist
            required = ["ds", "y", "aqi_lag_24h",
                        "aqi_roll_mean_24h", "hour_sin", "is_rush_hour"]
            missing  = [c for c in required if c not in df.columns]
            if missing:
                print(f"  WARNING — missing columns: {missing}")
            else:
                print(f"  All required columns present ✓")

            # Show weather coverage
            for col in ["temperature_2m", "wind_speed_10m",
                        "relative_humidity_2m"]:
                if col in df.columns:
                    pct = df[col].notna().mean() * 100
                    print(f"  {col}: {pct:.0f}% coverage")

            # Save features
            save_features(city, df)

            # Compute baselines
            stats = compute_baseline_stats(city, df)
            print(f"  Baseline saved ✓")

        except Exception as e:
            print(f"  ERROR: {e}")
            logger.error(f"Pipeline failed for {city}", exc_info=True)

    print(f"\n{'='*50}")
    print("Feature pipeline complete.")
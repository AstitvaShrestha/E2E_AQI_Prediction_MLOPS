"""
scripts/seed_from_kaggle.py

Seeds data/raw/ AND data/raw/{city}/weather/ from the
Kaggle dataset: Air Quality Dataset: Indian Cities (2022-2025)
(bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025)

This dataset already contains OpenMeteo weather merged with
PM2.5 readings — so one file gives us both AQI and weather
for the full training period 2022-2025.

For inference (live predictions), weather.py fetches fresh
OpenMeteo forecast data via API.
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CITIES, pm25_to_aqi

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SEED_DIR = DATA_DIR / "kaggle_seed"

CITY_VARIANTS = {
    "Delhi":     ["Delhi", "delhi", "New Delhi"],
    "Mumbai":    ["Mumbai", "mumbai"],
    "Kolkata":   ["Kolkata", "kolkata", "Calcutta"],
    "Chennai":   ["Chennai", "chennai"],
    "Bengaluru": ["Bengaluru", "bengaluru", "Bangalore"],
}

# ── Column mapping ─────────────────────────────────────────────────────────
# Maps Kaggle column names → our standard internal names
# Left  = Kaggle column name (from the dataset description)
# Right = our internal column name used by Spark pipeline + trainer

AQI_COLS = {
    "Datetime":      "timestamp",
    "City":          "city_raw",
    "PM2_5_ugm3":    "pm25",
}

WEATHER_COLS = {
    "Temp_2m_C":          "temperature_2m",       # °C
    "Wind_Speed_10m_kmh": "wind_speed_10m",        # km/h → convert to m/s
    "Wind_Dir_10m":       "wind_direction_10m",    # degrees
    "Humidity_Percent":   "relative_humidity_2m",  # %
    "Precipitation_mm":   "precipitation",         # mm
    "Pressure_MSL_hPa":   "surface_pressure",      # hPa
}

# Bonus India-specific features — use if available
BONUS_COLS = {
    "Temp_Inversion":      "temp_inversion",       # 0/1
    "Crop_Burning_Season": "crop_burning",         # 0/1
    "Festival_Period":     "festival_period",      # 0/1
    "Season":              "season",               # string
}


def find_csv() -> str:
    """Find the main Kaggle CSV file."""
    patterns = [
        str(SEED_DIR / "**/*.csv"),
        str(SEED_DIR / "*.csv"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            # Pick the largest CSV — likely the main data file
            return max(files, key=os.path.getsize)
    raise FileNotFoundError(
        f"No CSV found in {SEED_DIR}\n"
        "Download with:\n"
        "kaggle datasets download "
        "-d bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025 "
        f"-p {SEED_DIR}"
    )


def load_dataset() -> pd.DataFrame:
    """Load the Kaggle CSV and normalise column names."""
    csv_path = find_csv()
    print(f"Loading: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path)/1e6:.1f} MB\n")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}\n")

    # Verify required columns exist
    required = list(AQI_COLS.keys()) + list(WEATHER_COLS.keys())
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"WARNING — missing expected columns: {missing}")
        print("Available columns:", df.columns.tolist())

    return df


def extract_aqi(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Extract and clean AQI data for one city.
    Returns DataFrame with columns matching openaq.py output format:
    timestamp | pm25 | pm25_min | pm25_max | pm25_sd |
    coverage_pct | aqi | city | location_id
    """
    variants = [v.lower() for v in CITY_VARIANTS[city]]
    mask     = df["City"].str.lower().str.strip().isin(variants)
    city_df  = df[mask].copy()

    if city_df.empty:
        return pd.DataFrame()

    # Parse timestamp
    city_df["timestamp"] = pd.to_datetime(
        city_df["Datetime"], errors="coerce", utc=True
    )
    city_df = city_df.dropna(subset=["timestamp"])

    # Parse PM2.5
    city_df["pm25"] = pd.to_numeric(
        city_df["PM2_5_ugm3"], errors="coerce"
    )
    city_df = city_df.dropna(subset=["pm25"])
    city_df = city_df[
        (city_df["pm25"] >= 0) &
        (city_df["pm25"] <= 999)
    ]

    # Compute AQI from PM2.5 using our EPA formula
    # Note: dataset has US_AQI column but we use India CPCB standard
    # from config.py for consistency across the whole pipeline
    city_df["aqi"]          = city_df["pm25"].apply(pm25_to_aqi).astype(int)
    city_df["pm25_min"]     = city_df["pm25"]
    city_df["pm25_max"]     = city_df["pm25"]
    city_df["pm25_sd"]      = 0.0
    city_df["coverage_pct"] = 100.0
    city_df["city"]         = city
    city_df["location_id"]  = 0     # 0 = Kaggle seed

    out = city_df[[
        "timestamp", "pm25", "pm25_min", "pm25_max",
        "pm25_sd", "coverage_pct", "aqi", "city", "location_id"
    ]].sort_values("timestamp").drop_duplicates("timestamp")

    return out.reset_index(drop=True)


def extract_weather(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Extract weather data for one city from the same Kaggle CSV.
    Converts units where needed and renames to internal names.
    Returns DataFrame matching weather.py output format:
    timestamp | temperature_2m | wind_speed_10m | wind_direction_10m |
    relative_humidity_2m | precipitation | surface_pressure | city
    + bonus columns where available
    """
    variants = [v.lower() for v in CITY_VARIANTS[city]]
    mask     = df["City"].str.lower().str.strip().isin(variants)
    city_df  = df[mask].copy()

    if city_df.empty:
        return pd.DataFrame()

    # Parse timestamp
    city_df["timestamp"] = pd.to_datetime(
        city_df["Datetime"], errors="coerce", utc=True
    )
    city_df = city_df.dropna(subset=["timestamp"])

    # Build weather output DataFrame
    weather = pd.DataFrame()
    weather["timestamp"] = city_df["timestamp"]
    weather["city"]      = city

    # Map and convert each weather column
    for kaggle_col, internal_col in WEATHER_COLS.items():
        if kaggle_col not in city_df.columns:
            weather[internal_col] = None
            continue

        vals = pd.to_numeric(city_df[kaggle_col], errors="coerce")

        # Unit conversion: wind speed km/h → m/s
        # OpenMeteo API returns m/s but this dataset uses km/h
        # Prophet regressor values must be consistent between
        # training (Kaggle) and inference (OpenMeteo API)
        if kaggle_col == "Wind_Speed_10m_kmh":
            vals = vals / 3.6   # km/h ÷ 3.6 = m/s

        weather[internal_col] = vals

    # Add bonus India-specific columns if present
    for kaggle_col, internal_col in BONUS_COLS.items():
        if kaggle_col in city_df.columns:
            weather[internal_col] = city_df[kaggle_col].values

    weather = (weather
               .sort_values("timestamp")
               .drop_duplicates("timestamp")
               .reset_index(drop=True))

    return weather


def save_aqi_parquets(aqi_df: pd.DataFrame, city: str) -> int:
    """Save AQI data as daily Parquet files under data/raw/{city}/"""
    city_key = city.lower().replace(" ", "_")
    out_dir  = DATA_DIR / "raw" / city_key
    out_dir.mkdir(parents=True, exist_ok=True)

    records_saved = 0
    for date, group in aqi_df.groupby(aqi_df["timestamp"].dt.date):
        out_path = out_dir / f"date={date}.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            group    = (pd.concat([existing, group])
                        .drop_duplicates("timestamp")
                        .sort_values("timestamp")
                        .reset_index(drop=True))

        group.to_parquet(out_path, index=False)
        records_saved += len(group)

    return records_saved


def save_weather_parquets(weather_df: pd.DataFrame, city: str) -> int:
    """Save weather data as daily Parquet files under data/raw/{city}/weather/"""
    city_key    = city.lower().replace(" ", "_")
    weather_dir = DATA_DIR / "raw" / city_key / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    records_saved = 0
    for date, group in weather_df.groupby(
        weather_df["timestamp"].dt.date
    ):
        out_path = weather_dir / f"date={date}.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            group    = (pd.concat([existing, group])
                        .drop_duplicates("timestamp")
                        .sort_values("timestamp")
                        .reset_index(drop=True))

        group.to_parquet(out_path, index=False)
        records_saved += len(group)

    return records_saved


def verify():
    """Print summary of what was saved."""
    print(f"\n{'='*55}")
    print("Verification")
    print(f"{'='*55}")
    for city in CITIES:
        key         = city.lower().replace(" ", "_")
        aqi_files   = glob.glob(f"data/raw/{key}/date=*.parquet")
        wx_files    = glob.glob(f"data/raw/{key}/weather/date=*.parquet")

        if aqi_files:
            dates = sorted(
                f.split("date=")[1].replace(".parquet", "")
                for f in [Path(f).name for f in aqi_files]
            )
            print(f"  {city}:")
            print(f"    AQI:     {len(aqi_files):4d} files | "
                  f"{dates[0]} → {dates[-1]}")
        else:
            print(f"  {city}: NO AQI FILES")

        if wx_files:
            wdates = sorted(
                f.split("date=")[1].replace(".parquet","")
                for f in [Path(f).name for f in wx_files]
            )
            print(f"    Weather: {len(wx_files):4d} files | "
                  f"{wdates[0]} → {wdates[-1]}")
        else:
            print(f"    Weather: NO FILES")


def main():
    print("="*55)
    print("Seeding data/raw/ from Kaggle dataset")
    print("="*55 + "\n")

    # Load the full dataset once
    df = load_dataset()

    # Show cities available
    cities_in_data = df["City"].dropna().unique()
    print(f"Cities in dataset: {sorted(cities_in_data)}\n")

    total_aqi = 0
    total_wx  = 0

    for city in CITIES:
        print(f"\n{'─'*45}")
        print(f"Processing {city}...")
        print(f"{'─'*45}")

        # Extract AQI
        aqi_df = extract_aqi(df, city)
        if aqi_df.empty:
            print(f"  {city}: no AQI data found")
            continue

        aqi_saved = save_aqi_parquets(aqi_df, city)
        total_aqi += aqi_saved
        print(f"  AQI:    {aqi_saved:,} records saved")
        print(f"  Range:  {aqi_df['timestamp'].min().date()} → "
              f"{aqi_df['timestamp'].max().date()}")
        print(f"  PM2.5:  mean={aqi_df['pm25'].mean():.1f} "
              f"max={aqi_df['pm25'].max():.1f} µg/m³")
        print(f"  AQI:    mean={aqi_df['aqi'].mean():.0f} "
              f"max={aqi_df['aqi'].max()}")

        # Extract weather from same CSV
        wx_df = extract_weather(df, city)
        if wx_df.empty:
            print(f"  Weather: no data found")
        else:
            wx_saved   = save_weather_parquets(wx_df, city)
            total_wx  += wx_saved
            print(f"  Weather: {wx_saved:,} records saved")
            print(f"  Wind:   mean={wx_df['wind_speed_10m'].mean():.1f} m/s")
            print(f"  Temp:   mean={wx_df['temperature_2m'].mean():.1f} °C")

            # Show bonus columns that were saved
            bonus_found = [
                v for k, v in BONUS_COLS.items()
                if v in wx_df.columns
            ]
            if bonus_found:
                print(f"  Bonus columns: {bonus_found}")

    print(f"\n{'='*55}")
    print(f"Total AQI records:     {total_aqi:,}")
    print(f"Total weather records: {total_wx:,}")

    verify()

    print("\nNext steps:")
    print("  1. Run OpenAQ live fetch to append recent data")
    print("  2. Run weather.py to fetch recent OpenMeteo forecast")
    print("  3. Run src/features/pipeline.py to build features")


if __name__ == "__main__":
    main()
"""
OpenMeteo weather fetcher.
Free API — no key, no rate limits.

Two fetch modes:
  fetch_weather()            — archive + forecast split (for training gap fill)
  fetch_weather_last_n_days()— past_days parameter (for recent 7-day window)
  get_weather_forecast()     — future forecast for inference
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

try:
    from ..config import CITIES
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES

load_dotenv()

logger   = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
]


def _fetch_from_url(city: str, date_from: datetime,
                    date_to: datetime, base_url: str) -> pd.DataFrame:
    """Fetch weather from a specific OpenMeteo URL using date range."""
    cfg    = CITIES[city]
    params = {
        "latitude":   cfg["lat"],
        "longitude":  cfg["lon"],
        "hourly":     ",".join(WEATHER_VARS),
        "start_date": date_from.strftime("%Y-%m-%d"),
        "end_date":   date_to.strftime("%Y-%m-%d"),
        "timezone":   cfg["timezone"],
    }
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data   = resp.json()
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"], utc=False),
            **{v: hourly.get(v, [None] * len(hourly["time"]))
               for v in WEATHER_VARS}
        })
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_localize("Asia/Kolkata",
                            ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
        df["city"] = city
        return df.dropna(subset=["temperature_2m"]).reset_index(drop=True)

    except requests.RequestException as e:
        logger.error(f"OpenMeteo {base_url} failed for {city}: {e}")
        return pd.DataFrame()


def fetch_weather(city: str, date_from: datetime,
                  date_to: datetime) -> pd.DataFrame:
    """
    Fetch weather for a date range.
    Automatically routes to archive or forecast API based on dates.
    Use for training gap fill (large historical ranges).
    """
    now     = datetime.now(timezone.utc)
    cutoff  = now - timedelta(days=5)

    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=timezone.utc)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=timezone.utc)

    dfs = []

    if date_from < cutoff:
        hist_end = min(date_to, cutoff)
        df = _fetch_from_url(
            city, date_from, hist_end,
            "https://archive-api.open-meteo.com/v1/archive"
        )
        if not df.empty:
            dfs.append(df)

    if date_to > cutoff:
        fc_start = max(date_from, cutoff)
        df = _fetch_from_url(
            city, fc_start, date_to,
            "https://api.open-meteo.com/v1/forecast"
        )
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return (pd.concat(dfs)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True))


def fetch_weather_last_n_days(city: str, days: int = 7) -> pd.DataFrame:
    """
    Fetch last N days of actual weather using past_days parameter.
    This is the CORRECT way to get recent historical weather from
    OpenMeteo — do NOT use start_date/end_date on forecast API
    for past dates.

    Used for:
      - Drift detection recent window
      - Filling recent weather gap before inference
    """
    cfg = CITIES[city]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      cfg["lat"],
        "longitude":     cfg["lon"],
        "hourly":        ",".join(WEATHER_VARS),
        "past_days":     days,
        "forecast_days": 1,
        "timezone":      cfg["timezone"],
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data   = resp.json()
        hourly = data.get("hourly", {})

        if not hourly or "time" not in hourly:
            logger.error(f"No hourly data from OpenMeteo for {city}")
            return pd.DataFrame()

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"], utc=False),
            **{v: hourly.get(v, [None] * len(hourly["time"]))
               for v in WEATHER_VARS}
        })

        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_localize("Asia/Kolkata",
                            ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
        df["city"] = city
        df = df.dropna(subset=["temperature_2m"])

        # Keep only past — not the 1-day future forecast
        now = pd.Timestamp.now(tz="UTC")
        df  = df[df["timestamp"] <= now]

        logger.info(
            f"OpenMeteo {city}: {len(df)} rows "
            f"{df['timestamp'].min().date()} → "
            f"{df['timestamp'].max().date()}"
        )
        return df.sort_values("timestamp").reset_index(drop=True)

    except requests.RequestException as e:
        logger.error(
            f"OpenMeteo past_days failed for {city}: {e}"
        )
        return pd.DataFrame()


def get_weather_forecast(city: str, periods: int = 24) -> pd.DataFrame:
    """
    Get weather forecast for next N hours.
    Called by FastAPI at inference time.
    """
    cfg = CITIES[city]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      cfg["lat"],
        "longitude":     cfg["lon"],
        "hourly":        ",".join(WEATHER_VARS),
        "forecast_days": 3,
        "timezone":      cfg["timezone"],
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data   = resp.json()
        hourly = data.get("hourly", {})

        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"], utc=False),
            **{v: hourly.get(v, [None] * len(hourly["time"]))
               for v in WEATHER_VARS}
        })
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_localize("Asia/Kolkata",
                            ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
        df["city"] = city
        df = df.dropna(subset=["temperature_2m"])

        # Keep only future timestamps
        now = pd.Timestamp.now(tz="UTC")
        df  = df[df["timestamp"] >= now].head(periods)

        return df.reset_index(drop=True)

    except requests.RequestException as e:
        logger.error(f"OpenMeteo forecast failed for {city}: {e}")
        return pd.DataFrame()


def save_weather(city: str, df: pd.DataFrame) -> None:
    """Save weather data as daily Parquet files."""
    if df.empty:
        logger.warning(f"No weather data to save for {city}")
        return

    city_key    = city.lower().replace(" ", "_")
    weather_dir = DATA_DIR / "raw" / city_key / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for date, group in df.groupby(df["timestamp"].dt.date):
        out_path = weather_dir / f"date={date}.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            group    = (pd.concat([existing, group])
                        .drop_duplicates(subset=["timestamp"])
                        .sort_values("timestamp")
                        .reset_index(drop=True))
        group.to_parquet(out_path, index=False)
        saved += len(group)

    logger.info(f"Saved {saved} weather records for {city}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    print("Fetching last 7 days of weather for all cities...\n")

    for city in CITIES:
        # Use past_days parameter — correct method for recent data
        df = fetch_weather_last_n_days(city, days=7)
        if not df.empty:
            save_weather(city, df)
            print(f"{city}: {len(df)} records | "
                  f"{df['timestamp'].min().date()} → "
                  f"{df['timestamp'].max().date()}")
        else:
            print(f"{city}: FAILED")
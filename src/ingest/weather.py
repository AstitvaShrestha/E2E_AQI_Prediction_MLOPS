"""
OpenMeteo weather fetcher.
Free API — no key, no rate limits.
Historical archive: https://archive-api.open-meteo.com
Forecast:          https://api.open-meteo.com
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

from src.config import CITIES

load_dotenv()

logger   = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# These are the weather variables we request from OpenMeteo.
# Each one maps to a column in the output DataFrame.
WEATHER_VARS = [
    "temperature_2m",           # °C at 2 metres above ground
    "wind_speed_10m",           # m/s at 10 metres
    "wind_direction_10m",       # degrees 0-360
    "relative_humidity_2m",     # % at 2 metres
    "precipitation",            # mm per hour
    "surface_pressure",         # hPa — affects pollution dispersion
]

def fetch_weather(city, date_from, date_to):
    """
    Automatically splits request across archive + forecast APIs
    if date range spans the 5-day cutoff boundary.
    """
    now     = datetime.now(timezone.utc)
    cutoff  = now - timedelta(days=5)

    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=timezone.utc)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=timezone.utc)

    dfs = []

    # Historical part — use archive API
    if date_from < cutoff:
        hist_end = min(date_to, cutoff)
        df_hist  = _fetch_from_url(
            city, date_from, hist_end,
            "https://archive-api.open-meteo.com/v1/archive"
        )
        if not df_hist.empty:
            dfs.append(df_hist)

    # Forecast part — use forecast API
    if date_to > cutoff:
        fc_start = max(date_from, cutoff)
        df_fc    = _fetch_from_url(
            city, fc_start, date_to,
            "https://api.open-meteo.com/v1/forecast"
        )
        if not df_fc.empty:
            dfs.append(df_fc)

    if not dfs:
        return pd.DataFrame()

    combined = (pd.concat(dfs)
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True))
    return combined


def _fetch_from_url(city, date_from, date_to, base_url):
    """Internal — fetch weather from a specific OpenMeteo URL."""
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
            **{v: hourly.get(v, [None]*len(hourly["time"]))
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
        return df.reset_index(drop=True)

    except requests.RequestException as e:
        logger.error(f"OpenMeteo {base_url} failed for {city}: {e}")
        return pd.DataFrame()

def save_weather(city: str, df: pd.DataFrame) -> None:
    """
    Save weather data partitioned by date.
    One Parquet file per day under data/raw/{city}/weather/
    Merges with existing files to handle overlapping fetches.
    """
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
            group = (pd.concat([existing, group])
                     .drop_duplicates(subset=["timestamp"])
                     .sort_values("timestamp")
                     .reset_index(drop=True))

        group.to_parquet(out_path, index=False)
        saved += len(group)

    logger.info(f"Saved {saved} weather records for {city}")


def get_weather_forecast(
    city:    str,
    periods: int = 24,
) -> pd.DataFrame:
    """
    Get weather forecast for the next N hours.
    Called by FastAPI at inference time to fill Prophet regressors.
    Returns DataFrame with future timestamps and weather values.
    """
    date_from = datetime.now(timezone.utc)
    date_to   = date_from + timedelta(hours=periods + 1)
    return fetch_weather(city, date_from, date_to)


# def backfill_weather(city: str, years: int = 2) -> int:
#     """
#     Fetch full historical weather matching the AQI backfill period.
#     Chunks into 90-day windows — OpenMeteo handles large date ranges
#     but smaller chunks are more reliable and easier to retry.
#     """
#     total     = 0
#     date_to   = datetime.now(timezone.utc)
#     date_from = date_to - timedelta(days=365 * years)
#     current   = date_from

#     while current < date_to:
#         chunk_end = min(current + timedelta(days=90), date_to)
#         logger.info(
#             f"Weather {city}: "
#             f"{current.date()} → {chunk_end.date()}"
#         )
#         df = fetch_weather(city, current, chunk_end)
#         if not df.empty:
#             save_weather(city, df)
#             total += len(df)
#         current = chunk_end

#     logger.info(f"Weather backfill complete for {city}: {total} rows")
#     return total


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s %(name)s: %(message)s"
#     )
#     for city in CITIES:
#         backfill_weather(city, years=2)
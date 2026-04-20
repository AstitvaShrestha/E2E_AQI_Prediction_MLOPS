"""
AQICN live AQI fetcher.
Free API — register at https://aqicn.org/api/ for token.

Data note:
  AQICN reports US EPA AQI scale.
  We convert directly US AQI scale -> India CPCB AQI using
  band-to-band linear interpolation — no PM2.5 back-calculation needed.
  Weather fields (t, h, w, p, dew) are real physical units.
  Pollutant fields (pm25, pm10, no2 etc) are US AQI sub-indices.

Used by:
  - Airflow hourly DAG  → append today's reading to data/raw/
  - FastAPI /forecast   → fill aqi_lag_24h regressor at inference time
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

try:
    from ..config import CITIES, pm25_to_aqi
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES, pm25_to_aqi

load_dotenv()

logger      = logging.getLogger(__name__)
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "demo")
DATA_DIR    = Path(os.getenv("DATA_DIR", "data"))
BASE_URL    = "https://api.waqi.info/feed"

# AQICN city slugs
CITY_SLUGS = {
    "Delhi":     "delhi",
    "Mumbai":    "mumbai",
    "Kolkata":   "kolkata",
    "Chennai":   "chennai",
    "Bengaluru": "bangalore",
}
# CITY_SLUGS = {
#     "Delhi":     "@28.6139,77.2090",   # use coordinates for consistency
#     "Mumbai":    "@19.0760,72.8777",
#     "Kolkata":   "@22.5726,88.3639",
#     "Chennai":   "@13.0827,80.2707",
#     "Bengaluru": "@12.9716,77.5946",
# }


def fetch_current_aqi(city: str) -> dict | None:
    """
    Fetch current AQI and weather for one city from AQICN.

    Returns dict:
      timestamp   — UTC datetime
      aqi         — India CPCB AQI
      aqi_us      — original US AQI from AQICN
      temperature — °C  (real physical value)
      humidity    — %   (real physical value)
      wind_speed  — m/s (real physical value)
      wind_dir    — degrees
      wind_gust   — m/s
      pressure    — hPa (real physical value)
      dew_point   — °C  (real physical value)
      city        — city name
      source      — "aqicn"

    Returns None if request fails or city not found.
    """
    slug = CITY_SLUGS.get(city)
    if not slug:
        logger.error(f"No AQICN slug configured for: {city}")
        return None

    try:
        resp = requests.get(
            f"{BASE_URL}/{slug}/",
            params={"token": AQICN_TOKEN},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("status") != "ok":
            logger.warning(
                f"AQICN {city}: status={payload.get('status')} "
                f"— {payload.get('data', 'no detail')}"
            )
            return None

        d    = payload["data"]
        iaqi = d.get("iaqi", {})

        # ── Timestamp ─────────────────────────────────────────────
        time_info = d.get("time", {})
        ts_str    = time_info.get("s", "")
        tz_str    = time_info.get("tz", "+05:30")
        try:
            ts = pd.to_datetime(f"{ts_str}{tz_str}", utc=True)
        except Exception:
            ts = datetime.now(timezone.utc)

        # ── AQI — direct US → India conversion ────────────────────
        us_aqi = d.get("aqi")
        if us_aqi is None:
            logger.warning(f"AQICN {city}: no AQI in response")
            return None

        india_aqi = int(us_aqi)

        # ── Weather — these are real physical units in iaqi ────────
        result = {
            "timestamp":   ts,
            "aqi":         india_aqi,                        # India CPCB
            "aqi_us":      int(us_aqi),                      # US scale
            "temperature": iaqi.get("t",   {}).get("v"),     # °C
            "humidity":    iaqi.get("h",   {}).get("v"),     # %
            "wind_speed":  iaqi.get("w",   {}).get("v"),     # m/s
            "wind_dir":    iaqi.get("wd",  {}).get("v"),     # degrees
            "wind_gust":   iaqi.get("wg",  {}).get("v"),     # m/s
            "pressure":    iaqi.get("p",   {}).get("v"),     # hPa
            "dew_point":   iaqi.get("dew", {}).get("v"),     # °C
            "city":        city,
            "source":      "aqicn",
        }

        logger.info(
            f"AQICN {city}: "
            f"US_AQI={us_aqi} → India_AQI={india_aqi} | "
            f"T={result['temperature']}°C "
            f"H={result['humidity']}% "
            f"W={result['wind_speed']}m/s"
        )
        return result

    except requests.RequestException as e:
        logger.error(f"AQICN request failed for {city}: {e}")
        return None


def save_current_reading(city: str, reading: dict) -> bool:
    """
    Append current AQICN reading to today's Parquet file.
    Output format matches openaq.py and seed_from_kaggle.py exactly
    so the Spark pipeline and trainer work unchanged.
    """
    city_key = city.lower().replace(" ", "_")
    city_dir = DATA_DIR / "raw" / city_key
    city_dir.mkdir(parents=True, exist_ok=True)

    ts       = reading["timestamp"]
    date_str = ts.strftime("%Y-%m-%d")
    out_path = city_dir / f"date={date_str}.parquet"

    # pm25 stored as aqi value — AQICN does not provide raw concentration
    # location_id = -1 marks this row as AQICN-sourced
    row = pd.DataFrame([{
        "timestamp":    ts,
        "pm25":         float(reading["aqi_us"]),  # US AQI as pm25 proxy
        "pm25_min":     float(reading["aqi_us"]),
        "pm25_max":     float(reading["aqi_us"]),
        "pm25_sd":      0.0,
        "coverage_pct": 100.0,
        "aqi":          int(reading["aqi"]),        # India CPCB AQI
        "city":         city,
        "location_id":  -1,
    }])

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = (pd.concat([existing, row])
                    .drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True))
    else:
        combined = row

    combined.to_parquet(out_path, index=False)
    logger.info(f"Saved AQICN reading for {city} → {out_path.name}")
    return True


def ingest_all_cities() -> dict:
    """
    Fetch and save current AQI for all 5 cities.
    Called by Airflow hourly DAG every hour.
    Returns dict of {city: india_aqi} for logging/monitoring.
    """
    results = {}
    for city in CITIES:
        reading = fetch_current_aqi(city)
        if reading:
            save_current_reading(city, reading)
            results[city] = reading["aqi"]
            logger.info(f"{city}: AQI={reading['aqi']} saved")
        else:
            results[city] = None
            logger.warning(f"{city}: ingestion failed")
    return results


def get_current_aqi_value(city: str) -> float | None:
    """
    Get current India CPCB AQI for one city.
    Used by FastAPI at inference time to populate aqi_lag_24h regressor.
    Does NOT save to disk — read-only.

    Fallback chain:
      1. AQICN live API
      2. Last known value from disk (most recent Parquet file)
      3. None (caller handles missing value)
    """
    # Primary — AQICN live
    reading = fetch_current_aqi(city)
    if reading:
        return float(reading["aqi"])

    # Fallback — last known from disk
    logger.warning(f"AQICN unavailable for {city} — reading from disk")
    city_key = city.lower().replace(" ", "_")
    files    = sorted(
        (DATA_DIR / "raw" / city_key).glob("date=*.parquet"),
        reverse=True,
    )
    for f in files[:7]:
        try:
            df = pd.read_parquet(f)
            if not df.empty:
                last = float(
                    df.sort_values("timestamp").iloc[-1]["aqi"]
                )
                logger.info(
                    f"Using last known AQI for {city}: {last} "
                    f"(from {f.name})"
                )
                return last
        except Exception:
            continue

    logger.error(f"No AQI available for {city} from any source")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    print(f"Token: {AQICN_TOKEN[:8]}...\n")
    print("=" * 50)

    for city in CITIES:
        reading = fetch_current_aqi(city)
        if reading:
            print(f"\n{city}:")
            print(f"  US AQI:    {reading['aqi_us']}")
            print(f"  India AQI: {reading['aqi']} <- used by pipeline")
            print(f"  Temp:      {reading['temperature']} °C")
            print(f"  Humidity:  {reading['humidity']} %")
            print(f"  Wind:      {reading['wind_speed']} m/s "
                  f"@ {reading['wind_dir']}°")
            print(f"  Pressure:  {reading['pressure']} hPa")
            print(f"  Dew point: {reading['dew_point']} °C")
            print(f"  Time:      {reading['timestamp']}")
        else:
            print(f"\n{city}: FAILED")

    print("\n" + "=" * 50)
    print("Testing get_current_aqi_value (used by FastAPI):")
    for city in CITIES:
        val = get_current_aqi_value(city)
        print(f"  {city}: {val}")
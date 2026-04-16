"""
OpenAQ v3 API ingestion — fetches hourly PM2.5 readings per city.
Called by Airflow hourly DAG and backfill DAG.
"""

import os
import sys
import logging
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

try:
    # Prefer local package import to avoid collisions with other workspace projects.
    from ..config import CITIES, pm25_to_aqi
except ImportError:
    # Fallback when running this file directly (python src/ingest/openaq.py).
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES, pm25_to_aqi

load_dotenv()

logger = logging.getLogger(__name__)

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
OPENAQ_BASE    = "https://api.openaq.org/v3"
DATA_DIR       = Path(os.getenv("DATA_DIR", "data"))


# ---Session with auto-retry---

def make_session() -> requests.Session:
    """
    Requests session with exponential backoff retry.
    Handles 429 rate limit and 5xx server errors automatically.
    backoff_factor=1 means waits: 1s, 2s, 4s, 8s, 16s between retries.
    """
    session = requests.Session()
    retry   = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


SESSION = make_session()


# ---Sensor ID lookup----

def get_pm25_sensor_id(location_id: int, headers: dict) -> int | None:
    """
    Given an OpenAQ location ID, return the sensor ID for PM2.5.
    One location has multiple sensors (PM10, NO2, etc) — we only want PM2.5.
    Returns None if no PM2.5 sensor exists at this location.
    """
    url = f"{OPENAQ_BASE}/locations/{location_id}/sensors"
    try:
        resp = SESSION.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        sensors = resp.json().get("results", [])
        for s in sensors:
            if s.get("parameter", {}).get("name", "").lower() == "pm25":
                return s["id"]
        logger.warning(f"No PM2.5 sensor at location {location_id}")
        return None
    except requests.RequestException as e:
        logger.error(f"Sensor lookup failed for location {location_id}: {e}")
        return None


#---Raw result parser----

def _parse_results(all_results: list) -> pd.DataFrame:
    """
    Parse raw API result list into a clean DataFrame.
    Applies three quality filters:
      - coverage_pct >= 50  (sensor was active for at least half the hour)
      - hasFlags == False   (CPCB has not flagged the reading)
      - value is not None   (sensor was not completely offline)
    Also validates PM2.5 is within physical range 0-999 µg/m³.
    """
    if not all_results:
        return pd.DataFrame()

    rows = []
    for r in all_results:
        try:
            # Quality filter 1 — data coverage within the hour
            coverage = r.get("coverage", {}).get("percentCoverage", 100)
            if coverage < 50:
                continue

            # Quality filter 2 — CPCB quality flags
            if r.get("flagInfo", {}).get("hasFlags", False):
                continue

            # Quality filter 3 — null value means sensor was offline
            value = r.get("value")
            if value is None:
                continue

            rows.append({
                "timestamp":    pd.to_datetime(
                                    r["period"]["datetimeTo"]["utc"],
                                    utc=True
                                ),
                "pm25":         float(value),
                "pm25_min":     r.get("summary", {}).get("min"),
                "pm25_max":     r.get("summary", {}).get("max"),
                "pm25_sd":      r.get("summary", {}).get("sd"),
                "coverage_pct": float(coverage),
            })

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed record: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Physical bounds — PM2.5 cannot be negative or above 999 µg/m³
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= 999)].copy()

    # Compute AQI from PM2.5 using EPA formula (from config.py)
    df["aqi"] = df["pm25"].apply(pm25_to_aqi)

    # Sort chronologically, remove any duplicates
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True))

    return df


#---Hourly fetch with pagination---

def fetch_hourly_readings(
    sensor_id: int,
    date_from: datetime,
    date_to:   datetime,
    headers:   dict,
    max_pages: int = 25,
) -> pd.DataFrame:
    """
    Fetch all hourly readings for one sensor between two datetimes.
    Handles pagination (1000 records per page max).
    Sleeps 1.1s between pages to respect 60 req/min rate limit.
    Uses a while loop (not for loop) so we can retry the same page
    on 429 without skipping ahead.
    """
    all_results = []
    page        = 1

    while page <= max_pages:
        url    = f"{OPENAQ_BASE}/sensors/{sensor_id}/hours"
        params = {
            "datetime_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":         1000,
            "page":          page,
        }

        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=30)

            # Handle 429 manually for explicit logging
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                logger.warning(
                    f"Rate limited on page {page} — waiting {wait}s"
                )
                time.sleep(wait)
                continue          # retry same page — do NOT increment page

            resp.raise_for_status()
            results = resp.json().get("results", [])

            if not results:
                break             # no more data — stop paginating

            all_results.extend(results)
            logger.debug(
                f"Sensor {sensor_id} page {page}: {len(results)} records"
            )

            # Sleep between pages — stay under 60 req/min rate limit
            time.sleep(1.1)

            if len(results) < 1000:
                break             # last page — fewer than max means no more

            page += 1             # only advance if page succeeded

        except requests.RequestException as e:
            logger.error(
                f"Request failed for sensor {sensor_id} page {page}: {e}"
            )
            break

    return _parse_results(all_results)


#----City-level ingestion----

def ingest_city(city: str, date_from: datetime, date_to: datetime) -> int:
    """
    Fetch PM2.5 for all monitoring stations in a city.
    Averages readings across stations for each hour.
    Saves one Parquet file per day under data/raw/{city}/date=YYYY-MM-DD.parquet
    Returns total number of records saved.
    """
    cfg      = CITIES[city]
    headers  = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
    city_key = city.lower().replace(" ", "_")
    city_dir = DATA_DIR / "raw" / city_key
    city_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for loc_id in cfg["openaq_location_ids"]:
        sensor_id = get_pm25_sensor_id(loc_id, headers)
        if sensor_id is None:
            continue

        df = fetch_hourly_readings(sensor_id, date_from, date_to, headers)
        if not df.empty:
            df["city"]        = city
            df["location_id"] = loc_id
            all_dfs.append(df)
            logger.info(f"{city} loc {loc_id}: {len(df)} records")
        else:
            logger.warning(f"{city} loc {loc_id}: no data returned")

    if not all_dfs:
        logger.error(f"No data for {city} ({date_from.date()} – {date_to.date()})")
        return 0

    # Average PM2.5 across stations at same timestamp
    # pm25_min = smallest reading across all stations that hour
    # pm25_max = largest reading across all stations that hour
    combined = (
        pd.concat(all_dfs)
        .groupby("timestamp")
        .agg(
            pm25         = ("pm25",         "mean"),
            pm25_min     = ("pm25_min",     "min"),   # true minimum
            pm25_max     = ("pm25_max",     "max"),   # true maximum
            pm25_sd      = ("pm25_sd",      "mean"),
            coverage_pct = ("coverage_pct", "mean"),
            city         = ("city",         "first"),
        )
        .reset_index()
    )

    combined["aqi"] = combined["pm25"].apply(pm25_to_aqi).astype(int)

    # Save one Parquet file per day
    records_saved = 0
    for date, group in combined.groupby(combined["timestamp"].dt.date):
        out_path = city_dir / f"date={date}.parquet"

        if out_path.exists():
            # File exists — merge to handle overlapping fetches
            existing = pd.read_parquet(out_path)
            group = (
                pd.concat([existing, group])
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

        
        group.to_parquet(out_path, index=False)
        records_saved += len(group)

    logger.info(f"Saved {records_saved} records for {city}")
    return records_saved


# #----Historical backfill----

# def backfill_city(city: str, years: int = 2) -> int:
#     """
#     One-time 2-year historical data fetch.
#     Chunks into 30-day windows so each chunk fits in one API page.
#     30 days x 24 hours = 720 records = well under 1000/page limit.
#     """
#     total     = 0
#     date_to   = datetime.now(timezone.utc)
#     date_from = date_to - timedelta(days=365 * years)
#     current   = date_from

#     while current < date_to:
#         chunk_end = min(current + timedelta(days=30), date_to)
#         logger.info(f"Backfilling {city}: {current.date()} → {chunk_end.date()}")
#         count   = ingest_city(city, current, chunk_end)
#         total  += count
#         current = chunk_end

#     logger.info(f"Backfill complete for {city}: {total} total records")
#     return total


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from datetime import datetime, timedelta, timezone

    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=3)

    for city in CITIES:
        logger.info(f"Testing live fetch for {city}...")
        count = ingest_city(city, date_from, date_to)
        logger.info(f"{city}: {count} records saved")
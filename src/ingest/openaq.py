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
    from ..config import CITIES, pm25_to_aqi
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import CITIES, pm25_to_aqi

load_dotenv()

logger         = logging.getLogger(__name__)
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
OPENAQ_BASE    = "https://api.openaq.org/v3"
DATA_DIR       = Path(os.getenv("DATA_DIR", "data"))


def make_session() -> requests.Session:
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


def get_pm25_sensor_id(location_id: int, headers: dict) -> int | None:
    """
    Return the ACTIVE PM2.5 sensor at a location.
    Picks highest sensor ID = most recently registered = active sensor.
    Legacy sensors have low IDs (< 100,000).
    Active sensors have high IDs (> 12,000,000).
    """
    url = f"{OPENAQ_BASE}/locations/{location_id}/sensors"
    try:
        resp = SESSION.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        sensors = resp.json().get("results", [])

        pm25_sensors = [
            s for s in sensors
            if s.get("parameter", {}).get("name", "").lower() == "pm25"
        ]

        if not pm25_sensors:
            logger.warning(f"No PM2.5 sensor at location {location_id}")
            return None

        # Highest ID = most recently registered = active one
        best = max(pm25_sensors, key=lambda s: s["id"])
        logger.debug(
            f"Location {location_id}: picked sensor {best['id']} "
            f"from {len(pm25_sensors)} PM2.5 sensors"
        )
        return best["id"]

    except requests.RequestException as e:
        logger.error(f"Sensor lookup failed for location {location_id}: {e}")
        return None


def get_latest_reading(sensor_id: int, headers: dict) -> dict | None:
    """
    Fetch the single most recent hourly reading for a sensor.
    Uses sort_order=desc to get newest first.
    Returns dict with value and timestamp, or None if no data.
    """
    url    = f"{OPENAQ_BASE}/sensors/{sensor_id}/hours"
    params = {
        "limit":      1,
        "order_by":   "datetime",
        "sort_order": "desc",
    }
    try:
        resp = SESSION.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        r = results[0]
        return {
            "value":     r.get("value"),
            "timestamp": r["period"]["datetimeTo"]["utc"],
            "coverage":  r.get("coverage", {}).get("percentCoverage", 0),
        }
    except requests.RequestException as e:
        logger.error(f"Latest reading failed for sensor {sensor_id}: {e}")
        return None


def _parse_results(all_results: list) -> pd.DataFrame:
    if not all_results:
        return pd.DataFrame()

    rows = []
    for r in all_results:
        try:
            coverage = r.get("coverage", {}).get("percentCoverage", 100)
            if coverage < 50:
                continue
            if r.get("flagInfo", {}).get("hasFlags", False):
                continue
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
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= 999)].copy()
    df["aqi"] = df["pm25"].apply(pm25_to_aqi)
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True))
    return df


def fetch_hourly_readings(
    sensor_id: int,
    date_from: datetime,
    date_to:   datetime,
    headers:   dict,
    max_pages: int = 25,
) -> pd.DataFrame:
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

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited on page {page} — waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            results = resp.json().get("results", [])

            if not results:
                break

            all_results.extend(results)
            time.sleep(1.1)

            if len(results) < 1000:
                break

            page += 1

        except requests.RequestException as e:
            logger.error(f"Request failed for sensor {sensor_id} page {page}: {e}")
            break

    return _parse_results(all_results)


def ingest_city(city: str, date_from: datetime, date_to: datetime) -> int:
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

    combined = (
        pd.concat(all_dfs)
        .groupby("timestamp")
        .agg(
            pm25         = ("pm25",         "mean"),
            pm25_min     = ("pm25_min",     "min"),
            pm25_max     = ("pm25_max",     "max"),
            pm25_sd      = ("pm25_sd",      "mean"),
            coverage_pct = ("coverage_pct", "mean"),
            city         = ("city",         "first"),
        )
        .reset_index()
    )
    combined["aqi"] = combined["pm25"].apply(pm25_to_aqi).astype(int)

    records_saved = 0
    for date, group in combined.groupby(combined["timestamp"].dt.date):
        out_path = city_dir / f"date={date}.parquet"
        if out_path.exists():
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


def check_live_data(city: str) -> None:
    """
    Test-only function — checks latest available reading for each
    location in a city WITHOUT saving anything to disk.
    Shows sensor ID, latest timestamp, PM2.5 value, and data age.
    """
    cfg     = CITIES[city]
    headers = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
    now     = datetime.now(timezone.utc)

    print(f"\n{'='*55}")
    print(f"  {city} — {len(cfg['openaq_location_ids'])} locations")
    print(f"{'='*55}")

    if not cfg["openaq_location_ids"]:
        print("  No location IDs configured")
        return

    working = 0
    for loc_id in cfg["openaq_location_ids"]:
        sensor_id = get_pm25_sensor_id(loc_id, headers)
        if sensor_id is None:
            print(f"  Loc {loc_id}: no PM2.5 sensor")
            continue

        reading = get_latest_reading(sensor_id, headers)
        if not reading:
            print(f"  Loc {loc_id}: sensor {sensor_id} — NO DATA")
            continue

        ts_str   = reading["timestamp"]
        value    = reading["value"]
        coverage = reading["coverage"]

        # Calculate how old the data is
        ts       = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        age_hrs  = (now - ts).total_seconds() / 3600

        if value is not None:
            aqi = pm25_to_aqi(float(value))
            print(f"  Loc {loc_id}: sensor {sensor_id}")
            print(f"    PM2.5:    {value:.1f} µg/m³")
            print(f"    AQI:      {aqi}  (India CPCB)")
            print(f"    Latest:   {ts_str}")
            print(f"    Data age: {age_hrs:.1f} hours ago")
            print(f"    Coverage: {coverage:.0f}%")
            working += 1
        else:
            print(f"  Loc {loc_id}: sensor {sensor_id} — null value")

        time.sleep(0.5)

    print(f"\n  Summary: {working}/{len(cfg['openaq_location_ids'])} "
          f"locations have data")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,   # suppress INFO noise during test
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    print("OpenAQ live data check — no data saved to disk\n")
    print(f"Checking at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    for city in CITIES:
        check_live_data(city)

    print(f"\n{'='*55}")
    print("Done. Run ingest_city() to save data.")
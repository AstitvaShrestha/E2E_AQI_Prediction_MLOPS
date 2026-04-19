"""
Airflow DAG — one-time or on-demand historical data backfill.
Fetches AQI + weather for a specified date range.

Use cases:
  1. Initial setup — seed OpenAQ data before Kaggle period
  2. Gap fill — recover data after sensor/API outage
  3. New city onboarding — fetch historical data for a new city

Trigger manually from Airflow UI with conf:
  {
    "cities":    ["Delhi", "Mumbai"],   # optional — defaults to all
    "date_from": "2025-11-27",          # required
    "date_to":   "2026-04-10"           # optional — defaults to today
  }

NOT scheduled — manual trigger only.
"""


from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import logging
import pendulum


logger = logging.getLogger(__name__)

# --------Default args---------

DEFAULT_ARGS = {
    "owner":            "aqi_prediction",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
}


# ---Task Functions----

def validate_and_parse_conf(**context):
    """
    Parse and validate DAG conf parameters.
    Ensures date_from is provided and valid.
    Pushes parsed parameters to XCom.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CITIES
    from datetime import timezone


    conf = context["dag_run"].conf or {} 

    # date_from is required
    date_from_str = conf.get("date_from")

    if not date_from_str:
        raise ValueError(
            "date_from is required in conf. "
            "Example: {'date_from': '2025-11-27'}"
        )
    
    try:
        date_from = datetime.strptime(
            date_from_str, "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc)
    
    except ValueError:
        raise ValueError(
            f"Invalid date_from format: '{date_from_str}'. "
            "Expected: YYYY-MM-DD"
        )
    
    # date_to is optional, defaults to today
    date_to_str = conf.get("date_to")

    if date_to_str:
        try:
            date_to = datetime.strptime(
                date_to_str, "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)

        except ValueError:
            raise ValueError(
                f"Invalid date_to format: '{date_to_str}'. "
                "Expected: YYYY-MM-DD"
            )
    else:
        date_to = datetime.now(timezone.utc)

    # Validate date range
    if date_from >= date_to:
         raise ValueError(
            f"date_from ({date_from_str}) must be before "
            f"date_to ({date_to.date()})"
        )
    
    max_days = (date_to - date_from).days

    if max_days > 730: # ~2 years of hourly data
         raise ValueError(
            f"Date range too large: {max_days} days. "
            "Max allowed: 730 days (2 years). "
            "Split into smaller ranges."
        )
    

    # cities is optional, defaults to all
    cities_conf = conf.get("cities", list(CITIES.keys()))
    valid_cities = [c for c in cities_conf if c in CITIES]
    invalid      = [c for c in cities_conf if c not in CITIES]

    if invalid:
        logger.warning(f"Unknown cities skipped: {invalid}")

    if not valid_cities:
        raise ValueError(
            f"No valid cities. "
            f"Received: {cities_conf}. "
            f"Valid: {list(CITIES.keys())}"
        )

    params = {
        "date_from":    date_from.isoformat(),
        "date_to":      date_to.isoformat(),
        "cities":       valid_cities,
        "days":         max_days,
    }

    logger.info(
        f"Backfill params: "
        f"{date_from.date()} → {date_to.date()} "
        f"({max_days} days) | "
        f"cities: {valid_cities}"
    )


    context["ti"].xcom_push(keys="params", value=params)
    return params


def backfill_openaq(**context):
    """
    Fetch historical AQI from OpenAQ for the specified date range.
    Chunks into 30-day windows to avoid API timeouts.
    Merges with existing data — safe to rerun.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from datetime import timezone
    from src.ingest.openaq import ingest_city


    ti = context["ti"]
    params = ti.xcom_pull(task_ids="validate_conf", keys="params")

    date_from = datetime.fromisoformat(params["date_from"])
    date_to   = datetime.fromisoformat(params["date_to"])
    cities    = params["cities"]

    results = {}

    for city in cities:
        logger.info(
            f"Backfilling OpenAQ {city}: "
            f"{date_from.date()} → {date_to.date()}"
        )

        city_total = 0
        current    = date_from

        # Chunk into 30-day windows
        # 30 days × 24 hours = 720 records — fits in one API page
        while current < date_to:
            chunk_end = min(current + timedelta(days=30), date_to)

            try:
                count = ingest_city(city, current, chunk_end)
                city_total += count
                logger.info(
                    f"  {city} chunk "
                    f"{current.date()} → {chunk_end.date()}: "
                    f"{count} records"
                )

            except Exception as e:
                logger.error(
                    f"  {city} chunk "
                    f"{current.date()} → {chunk_end.date()} "
                    f"failed: {e}"
                )
            
            current = chunk_end

        results[city] = city_total
        logger.info(f"OpenAQ backfill {city}: {city_total} total records")


    ti.xcom_push(key="openaq_counts", value=results)
    return results


def backfill_weather(**context):
    """
    Fetch historical weather from OpenMeteo for the specified date range.
    OpenMeteo archive API handles large date ranges reliably.
    No chunking needed — OpenMeteo handles multi-year requests.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.ingest.weather import fetch_weather, save_weather

    ti     = context["ti"]
    params = ti.xcom_pull(task_ids="validate_conf", key="params")

    date_from = datetime.fromisoformat(params["date_from"])
    date_to   = datetime.fromisoformat(params["date_to"])
    cities    = params["cities"]

    results = {}

    for city in cities:

        try:
            df = fetch_weather(city, date_from, date_to)

            if not df.empty:
                save_weather(city, df)
                results[city] = len(df)
                logger.info(
                    f"Weather backfill {city}: "
                    f"{len(df)} records | "
                    f"{df['timestamp'].min().date()} → "
                    f"{df['timestamp'].max().date()}"
                )
            else:
                results[city] = 0
                logger.warning(
                    f"Weather backfill {city}: no data returned"
                )

        except Exception as e:
            logger.error(
                f"Weather backfill failed for {city}: {e}"
            )
            results[city] = 0

    ti.xcom_push(key="weather_counts", value=results)
    return results


def log_backfill_summary(**context):
    """
    Log summary of backfill results.
    Shows records fetched per city per source.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CITIES

    ti     = context["ti"]
    params = ti.xcom_pull(task_ids="validate_conf", key="params")

    openaq_counts  = ti.xcom_pull(
        task_ids="backfill_openaq",  key="openaq_counts"
    ) or {}
    weather_counts = ti.xcom_pull(
        task_ids="backfill_weather", key="weather_counts"
    ) or {}

    cities    = params["cities"]
    date_from = params["date_from"][:10]
    date_to   = params["date_to"][:10]
    days      = params["days"]
    expected  = days * 24   # expected hours

    logger.info("=" * 60)
    logger.info(f"BACKFILL SUMMARY: {date_from} → {date_to} ({days} days)")
    logger.info("=" * 60)
    logger.info(
        f"{'City':<12} {'OpenAQ':>10} {'Weather':>10} "
        f"{'AQI Coverage':>14}"
    )
    logger.info("-" * 60)


    for city in cities:
        openaq_n = openaq_counts.get(city, 0)
        weather_n = weather_counts.get(city, 0)
        coverage  = f"{openaq_n/expected*100:.1f}%" if expected > 0 else "N/A"


        logger.info(
            f"{city:<12} "
            f"{openaq_n:>10} "
            f"{weather_n:>10} "
            f"{coverage:>14}"
        )

    logger.info("=" * 60)
    logger.info(
        "Next steps: run pipeline.py then trainer.py "
        "to rebuild features and retrain models."
    )


# --- DAG Definition ---
with DAG(
    dag_id = "aqi_backfill",
    default_args = DEFAULT_ARGS,
    description = "On-demand historical AQI + weather backfill",
    schedule = None,        # manual trigger only
    start_date = pendulum.now().subtract(days=1),
    catchup = False,
    max_active_runs = 1,
    tags = ["aqi_prediction", "backfill", "ingestion"],
) as dag:
    
    t_validate = PythonOperator(
        task_id           = "validate_conf",
        python_callable   = validate_and_parse_conf,
        execution_timeout = timedelta(minutes=2),
    )

    t_openaq = PythonOperator(
        task_id           = "backfill_openaq",
        python_callable   = backfill_openaq,
        execution_timeout = timedelta(hours=6),  # large range can be slow
    )

    t_weather = PythonOperator(
        task_id           = "backfill_weather",
        python_callable   = backfill_weather,
        execution_timeout = timedelta(hours=2),
    )

    t_summary = PythonOperator(
        task_id           = "log_summary",
        python_callable   = log_backfill_summary,
        execution_timeout = timedelta(minutes=5),
        trigger_rule      = "all_done",
    )

    # Dependencies
    #
    # validate_conf
    #       │
    #       ├──> backfill_openaq  ──┐
    #       └──> backfill_weather ──┴──> log_summary
    #
    # OpenAQ and weather fetch run in parallel after validation

    t_validate >> [t_openaq, t_weather] >> t_summary


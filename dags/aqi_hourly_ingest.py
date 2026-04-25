"""
Airflow DAG — runs every hour to:
  1. Fetch latest AQI from OpenAQ for all 5 cities
  2. Fetch latest weather from OpenMeteo for all 5 cities
  3. Append current AQI from AQICN (single reading per city)
  4. Run drift detection
  5. Trigger retraining DAG if drift detected

Schedule: every hour at :05 past the hour
  (5 minute offset avoids top-of-hour API congestion)
"""

import requests
import os
from datetime import datetime, timedelta
import pendulum


from airflow import DAG
from airflow.providers.standard.operators.python import (
    PythonOperator,
    BranchPythonOperator,
)
from airflow.providers.standard.operators.empty import EmptyOperator
import logging

logger = logging.getLogger(__name__)


#---Default args-------
DEFAULT_ARGS = {
    "owner": "aqi_prediction",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

#---Task functions------

def ingest_openaq(**context):
    """
    Fetch last 2 hours of AQI from OpenAQ for all cities.
    Uses 2-hour window to handle slight API delays.
    Saves to data/raw/{city}/date=YYYY-MM-DD.parquet
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from datetime import datetime, timedelta, timezone
    from src.ingest.openaq import ingest_city
    from src.config import CITIES

    
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(hours=2)

    results = {}

    for city in CITIES:
        try:
            count = ingest_city(city, date_from, date_to)
            results[city] = count
            logger.info(f"OpenAQ {city}: {count} records")

        except Exception as e:
            logger.error(f"OpenAQ ingestion failed for {city}: {e}")
            results[city] = 0

    # Push results to XCom for downstream tasks
    context["ti"].xcom_push(key="openaq_counts", value=results)

    return results


def ingest_weather(**context):
    """
    Fetch last 2 hours of weather from OpenMeteo for all cities.
    Uses past_days=1 to get recent data reliably.
    Saves to data/raw/{city}/weather/date=YYYY-MM-DD.parquet
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.ingest.weather import fetch_weather_last_n_days, save_weather
    from src.config import CITIES


    results = {}
    for city in CITIES:
        try:
            df = fetch_weather_last_n_days(city, days=1)
            if not df.empty:
                save_weather(city, df)
                results[city] = len(df)
                logger.info(f"Weather {city}: {len(df)} records")

            else:
                results[city] = 0
                logger.warning(f"Weather {city}: no data returned")

        except Exception as e:
            logger.error(f"Weather ingestion failed for {city}: {e}")
            results[city] = 0
    
    context["ti"].xcom_push(key="weather_counts", value=results)
    return results


def ingest_aqicn(**context):
    """
    Fetch current hour AQI from AQICN for all cities.
    Appends single reading to today's raw file.
    Used as aqi_lag_24h regressor at inference time.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.ingest.aqicn import fetch_current_aqi, save_current_reading
    from src.config import CITIES

    results = {}
    for city in CITIES:
        try:
            reading = fetch_current_aqi(city)
            if reading:
                save_current_reading(city, reading)
                results[city] = reading["aqi"]
                logger.info(
                    f"AQICN {city}: AQI={reading['aqi']}"
                )

            else:
                results[city] = None
                logger.warning(f"AQICN {city}: no data")

        except Exception as e:
            logger.error(f"AQICN ingestion failed for {city}: {e}")
            results[city] = None

    context["ti"].xcom_push(key="aqicn_readings", value=results)
    
    return results


def run_drift_detection(**context):
    """
    Run KS-test drift detection for all cities.
    Pushes list of drifted cities to XCom.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.features.drift import check_all_cities, should_retrain


    results = check_all_cities()
    drifted_cities = should_retrain(results)

    logger.info(f"Drift check complete. Drifted: {drifted_cities}")


    for city, r in results.items():

        if r.get("error"):
            logger.warning(f"Drift check error for {city}: {r['error']}")
        
        else:
            logger.info(
                f"Drift {city}: p={r['p_value']:.4f} "
                f"KS={r['ks_stat']:.4f} "
                f"drifted={r['is_drifted']}"
            )

    context["ti"].xcom_push(key="drift_results", value=results)
    context["ti"].xcom_push(key="drifted_cities", value=drifted_cities)
    context["ti"].xcom_push(key="retrain_required", value=len(drifted_cities)>0)


    return drifted_cities


def check_retrain_needed(**context):
    """
    Branch operator — decides whether to trigger retraining.
    Returns task_id of next task to execute.
    """

    ti = context["ti"]
    drifted_cities = ti.xcom_pull(
        task_ids="drift_detection",
        key="drifted_cities"
    )

    if drifted_cities:
        logger.info(
            f"Drift detected in {drifted_cities} — triggering retrain"
        )
        return "trigger_retrain"
    
    else:
        logger.info("No drift — skipping retrain")
        return "no_retrain_needed"


def trigger_retrain_dag(**context):
    """
    Trigger the retraining DAG for drifted cities.
    Uses Airflow's TriggerDagRunOperator pattern via API.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


    ti = context["ti"]
    drifted_cities = ti.xcom_pull(
        task_ids="drift_detection",
        key="drifted_cities"
    )

    logger.info(f"Triggering retraining for: {drifted_cities}")

    # Trigger via Airflow REST API using JWT token
    from datetime import timezone
    airflow_url = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
    airflow_user = os.getenv("AIRFLOW_USERNAME", "admin")
    airflow_pass = os.getenv("AIRFLOW_PASSWORD", "admin")

    # Get JWT token
    token_resp = requests.post(
        f"{airflow_url}/auth/token",
        json={"username": airflow_user, "password": airflow_pass},
        timeout=10,
    )
    if token_resp.status_code not in (200, 201):
        logger.error(f"Failed to get auth token: {token_resp.status_code} {token_resp.text}")
        return drifted_cities

    token = token_resp.json()["access_token"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        now = datetime.now(tz=timezone.utc)
        resp = requests.post(
            f"{airflow_url}/api/v2/dags/aqi_retrain/dagRuns",
            json={
                "conf": {"cities": drifted_cities},
                "dag_run_id": f"drift_triggered_{now.strftime('%Y%m%d_%H%M%S')}",
                "logical_date": now.isoformat(),
            },
            headers=headers,
            timeout=30,
        )

        if resp.status_code in (200, 201, 409):
            logger.info(f"Retrain DAG triggered for: {drifted_cities}")
        else:
            logger.error(f"Failed to trigger retrain DAG: {resp.status_code} {resp.text}")

    except Exception as e:
        logger.error(f"Could not trigger retrain DAG: {e}")

    return drifted_cities





def log_ingestion_summary(**context):
    """
    Log summary of all ingestion tasks for monitoring.
    This data appears in Airflow task logs and Prometheus metrics.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ti = context["ti"]

    openaq_counts = ti.xcom_pull(
        task_ids="ingest_openaq", key="openaq_counts"
    ) or {}


    weather_counts = ti.xcom_pull( 
        task_ids="ingest_weather", key="weather_counts"
    ) or {}

    aqicn_readings = ti.xcom_pull(
        task_ids="ingest_aqicn", key="aqicn_readings"
    ) or {}

    drifted = ti.xcom_pull(
        task_ids="drift_detection", key="drifted_cities"
    ) or []

    logger.info("=" * 50)
    logger.info("HOURLY INGESTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"{'City':<12} {'OpenAQ':>8} {'Weather':>9} {'AQICN AQI':>10}")
    logger.info("-" * 50)

    from src.config import CITIES

    for city in CITIES:
        logger.info(
            f"{city:<12} "
            f"{openaq_counts.get(city, 0):>8} "
            f"{weather_counts.get(city, 0):>9} "
            f"{str(aqicn_readings.get(city, 'N/A')):>10}"
        )

    logger.info("-" * 50)
    logger.info(f"Drifted cities: {drifted if drifted else 'none'}")
    logger.info("=" * 50)


# ---DAG Definition------


with DAG(
    dag_id = "aqi_hourly_ingest",
    default_args = DEFAULT_ARGS,
    description = "Hourly AQI + weather ingestion and drift detection",
    schedule = "5 * * * *",  # every hour at :05
    start_date = pendulum.now().subtract(days=1),  # start yesterday to allow immediate run
    catchup = False, # don't backfill missed runs
    max_active_runs = 1, # prevent overlapping runs
    tags = ["aqi_prediction", "ingestion", "production"],
) as dag:

    # Tasks
    t_openaq = PythonOperator(
        task_id = "ingest_openaq",
        python_callable = ingest_openaq,
        execution_timeout = timedelta(minutes=30),
    )

    t_weather = PythonOperator(
        task_id = "ingest_weather",
        python_callable = ingest_weather,
        execution_timeout = timedelta(minutes=10),
    )

    t_aqicn = PythonOperator(
        task_id = "ingest_aqicn",
        python_callable = ingest_aqicn,
        execution_timeout = timedelta(minutes=5),
    )

    t_drift = PythonOperator(
        task_id         = "drift_detection",
        python_callable = run_drift_detection,
        execution_timeout = timedelta(minutes=10),
    )

    t_branch = BranchPythonOperator(
        task_id = "check_retrain",
        python_callable = check_retrain_needed,
    )

    t_retrain = PythonOperator(
        task_id = "trigger_retrain",
        python_callable = trigger_retrain_dag,
        execution_timeout = timedelta(minutes=5),
    )

    t_no_retrain = EmptyOperator(
        task_id = "no_retrain_needed"
    )


    t_summary = PythonOperator(
        task_id         = "log_summary",
        python_callable = log_ingestion_summary,
        trigger_rule    = "all_done",   # runs even if upstream fails
    )


    # Task dependencies
    #
    # OpenAQ ──┐
    # Weather ─┼──> Drift ──> Branch ──> Retrain ──> Summary
    # AQICN ───┘                    └──> No retrain ──┘
    #
    # Reload happens in aqi_retrain DAG after training completes

    [t_openaq, t_weather, t_aqicn] >> t_drift
    t_drift >> t_branch
    t_branch >> [t_retrain, t_no_retrain]
    [t_retrain, t_no_retrain] >> t_summary

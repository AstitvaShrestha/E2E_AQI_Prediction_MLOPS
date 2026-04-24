"""
Airflow DAG — triggered when drift is detected in aqi_hourly_ingest.
Retrains Prophet models for drifted cities and promotes if better.

Two trigger modes:
  1. Automatic — triggered by aqi_hourly_ingest when drift detected
  2. Manual    — trigger from Airflow UI with city list in conf

Example manual trigger conf:
  {"cities": ["Delhi", "Mumbai"]}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
import logging
import pendulum

logger = logging.getLogger(__name__)


# ----------Default args------------
DEFAULT_ARGS = {
    "owner": "aqi_prediction",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,           # only 1 retry — training is expensive
    "retry_delay":      timedelta(minutes=10),
}


#-------Task functions-------

def get_cities_to_retrain(**context):
    """
    Determine which cities need retraining.

    Two sources:
      1. DAG conf — when triggered by aqi_hourly_ingest
         conf = {"cities": ["Delhi", "Mumbai"]}
      2. All cities — when triggered manually without conf

    Pushes city list to XCom for downstream tasks.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CITIES

    # Get conf from DAG trigger
    conf = context.get("dag_run").conf or {}
    cities = conf.get("cities", list(CITIES.keys()))


    # Validate — ensure all cities are known
    valid_cities = [c for c in cities if c in CITIES]
    invalid = [c for c in cities if c not in CITIES]

    if invalid:
        logger.warning(f"Unknown cities in conf, skipping: {invalid}")

    if not valid_cities:
        logger.error("No valid cities to retrain")
        raise ValueError(
            f"No valid cities to retrain. "
            f"Received: {cities}. "
            f"Valid options: {list(CITIES.keys())}"
        )


    logger.info(f"Cities to retrain: {valid_cities}")
    context["ti"].xcom_push(key="cities", value=valid_cities)
    return valid_cities


def rebuild_features(**context):
    """
    Rebuild features.parquet for all cities before retraining.

    Why rebuild before retraining:
      - New data has been ingested since last training
      - Gap period may have been filled
      - Bonus features (crop_burning) need recalculation
      - Ensures model trains on freshest available data

    Uses pandas fallback if Spark unavailable.
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.features.pipeline import (
        build_features_spark,
        save_features,
        compute_baseline_stats,
    )
    from src.config import CITIES

    ti = context["ti"]
    cities = ti.xcom_pull(
        task_ids="get_cities",
        key = "cities",
    ) or list(CITIES.keys())


    results = {}

    for city in cities:
        try:
            df = build_features_spark(city)
            save_features(city, df)
            compute_baseline_stats(city, df)
            results[city] = len(df)

            logger.info(
                f"Features rebuilt for {city}: "
                f"{len(df)} rows | "
                f"{df['ds'].min().date()} → {df['ds'].max().date()}"
            )

        except Exception as e:
            logger.error(f"Feature rebuild failed for {city}: {e}")
            results[city] = 0

    ti.xcom_push(key="feature_counts", value=results)
    
    return results


def retrain_city(city, **context):
    """
    Retrain Prophet + SARIMA for one city.
    Promotes new model to champion if MAE improves.

    Separate task per city so:
      - Cities retrain in parallel (Airflow schedules them)
      - One city failing does not block others
      - Each city has its own retry count
    """

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import pandas as pd
    from src.train.trainer import (
        train_prophet,
        evaluate_and_promote,
    )

    feat_path = (
        Path("data/processed") / city.lower().replace(" ", "_") /
        "features.parquet"
    )

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features not found for {city}: {feat_path}. "
            "Run rebuild_features first."
        )


    df = pd.read_parquet(feat_path)

    logger.info(
        f"Retraining {city}: "
        f"{len(df)} rows | "
        f"{df['ds'].min().date()} → {df['ds'].max().date()}"
    )

    # Train Prophet
    model, prophet_metrics = train_prophet(city, df)
    logger.info(f"Prophet {city}: MAE={prophet_metrics['mae']:.2f}")


    # Promote if better than current champion
    promoted = evaluate_and_promote(
        city, 
        prophet_metrics["mae"],
        prophet_metrics["version"],
    )

    result = {
        "city":        city,
        "prophet_mae": prophet_metrics["mae"],
        "promoted":    promoted,
        "version":     prophet_metrics["version"],
    }

    logger.info(
        f"Retrain complete {city}: "
        f"Prophet MAE={prophet_metrics['mae']:.2f} | "
        f"{'PROMOTED ✓' if promoted else 'not promoted'}"
    )

    context["ti"].xcom_push(key=f"result_{city}", value=result)
    return result


def reload_api_models(**context):
    """
    Notify FastAPI to reload champion models after retraining.
    Must run after ALL city retraining tasks complete.
    """
    import requests
    import os

    api_url = os.getenv("FASTAPI_URL", "http://localhost:8000") 

    try:
        resp = requests.post(f"{api_url}/reload-models", timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                f"API models reloaded: "
                f"{data.get('models_loaded', [])}"
            )

        else:
            logger.warning(
                f"Model reload returned {resp.status_code}: "
                f"{resp.text}"
            )


    except Exception as e:
        logger.warning(f"Could not reload API models: {e}")



def log_retrain_summary(**context):
    """
    Log retraining summary — shows which cities improved
    and which were rejected.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CITIES

    ti = context["ti"]
    cities = ti.xcom_pull(
        task_ids="get_cities",
        key = "cities",
    ) or list(CITIES.keys())


    logger.info("=" * 60)
    logger.info("RETRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'City':<12} {'Prophet MAE':>12} {'Status'}")
    logger.info("-" * 40)

    promoted_count = 0

    for city in cities:
        result = ti.xcom_pull(
            task_ids=f"retrain_{city.lower()}",
            key=f"result_{city}",
        )

        if result:
            status = "champion ✓" if result["promoted"] else "rejected"

            if result["promoted"]:
                promoted_count += 1

            logger.info(
                f"{city:<12} "
                f"{result['prophet_mae']:>12.2f} "
                f"{status}"
            )

        else:
            logger.warning(f"{city:<12} FAILED")

    logger.info("-" * 40)
    logger.info(
        f"Promoted {promoted_count}/{len(cities)} models to champion"
    )
    logger.info("=" * 60)


#-------DAG definition-------

with DAG(
    dag_id = "aqi_retrain",
    default_args = DEFAULT_ARGS,
    description = "Retrain AQI models for drifted cities",
    schedule = None,        # triggered only — not on schedule
    start_date = pendulum.now().subtract(days=1),
    catchup = False,
    max_active_runs = 1,           # one retraining at a time
    tags = ["aqi_prediction", "training", "mlops"],
) as dag:
    
    # Tasks

    t_get_cities = PythonOperator(
        task_id = "get_cities",
        python_callable = get_cities_to_retrain,
        execution_timeout = timedelta(minutes=2),
    )

    t_rebuild = PythonOperator(
        task_id = "rebuild_features",
        python_callable = rebuild_features,
        execution_timeout = timedelta(minutes=30),
    )

    # One retrain task per city — run in parallel
    retrain_tasks = []

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CITIES

    for city in CITIES.keys():
        t = PythonOperator(
            task_id = f"retrain_{city.lower()}",
            python_callable = retrain_city,
            op_kwargs = {"city": city},
            execution_timeout = timedelta(hours=2),
            pool = "retrain_pool",  # limit concurrent retrains to 2
        )

        retrain_tasks.append(t)

    t_reload = PythonOperator(
        task_id = "reload_api_models",
        python_callable = reload_api_models,
        execution_timeout = timedelta(minutes=5),
        trigger_rule    = "all_done",  # reload even if some cities failed
    )

    t_summary = PythonOperator(
        task_id = "log_retrain_summary",
        python_callable = log_retrain_summary,
        execution_timeout = timedelta(minutes=5),
        trigger_rule = "all_done",
    )


    # Task dependencies
    #
    # get_cities ──> rebuild_features ──> retrain_Delhi   ──┐
    #                                  ──> retrain_Mumbai  ──┤
    #                                  ──> retrain_Kolkata ──┼──> reload_api ──> summary
    #                                  ──> retrain_Chennai ──┤
    #                                  ──> retrain_Bengaluru┘
    #
    # All 5 cities retrain in parallel after features are rebuilt

    t_get_cities >> t_rebuild >> retrain_tasks
    retrain_tasks >> t_reload >> t_summary
# AI Disclosure — AQI Prediction System


**Course:** Machine Learning Operations (MLOps)
**Project:** End-to-End MLOps  
**Date:** April 2026


## 1. Overview of AI Assistance

This project used Claude & ChatGPT as an AI coding assistant throughout development. This document discloses the AI contributions in accordance with academic integrity guidelines.

AI assistance was used for: code generation, debugging, architectural decisions, documentation writing, and code comments.


## 2. Key Queries and AI Responses

The following section documents significant queries made to Claude and the nature of responses received. Code shown is representative — full code is in the repository.


### 2.1 Airflow DAG Architecture

**Query:** "How to change this DAG to trigger only and not schedule and how to build task dynamically for only cities that require it "

**Code generated (example):**
```python
with DAG(
    dag_id            = "aqi_retrain",
    schedule_interval = None,        # triggered only — not on schedule
    start_date        = days_ago(1),
    catchup           = False,
    max_active_runs   = 1,
) as dag:
    retrain_tasks = []
    for city in ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]:
        t = PythonOperator(
            task_id         = f"retrain_{city.lower()}",
            python_callable = retrain_city,
            op_kwargs       = {"city": city},
            execution_timeout = timedelta(hours=2),
        )
        retrain_tasks.append(t)
    t_get_cities >> t_rebuild >> retrain_tasks
    retrain_tasks >> t_reload >> t_summary
```

---

### 2.2 Docker Compose and Dockerfile

**Query:** "Debug why in docker-compose.yaml authentication is not working and not able to call 2nd dag"

**AI Response:** Diagnosed that Airflow 3.x removed basic auth support on `/api/v2` — only JWT Bearer tokens are accepted. Fix: call `/auth/token` first to obtain a JWT, then use `Authorization: Bearer <token>` when triggering `aqi_retrain`. Also identified that SimpleAuthManager auto-generates a random password on each restart, making credentials unreliable — recommended migrating to FAB (`apache-airflow-providers-fab`) for stable `admin:admin` credentials.

---

### 2.3 Drift Detection Bug Fixes

**Query (paraphrased):** "why is p-value close to 0 even if AQI mean is not that different?"

**AI Response:** AI identified three distinct bugs in the drift detection implementation:

1. **`get_recent_window` used `drop_duplicates` instead of `groupby.mean()`** — this compared raw multi-station data against processed single-station baseline, creating an apples-to-oranges comparison.

2. **Sample size asymmetry** — the baseline had 2160 rows vs 122 recent rows, causing KS-test to always return p≈0 due to statistical power scaling with N. Fix: subsample baseline to `len(recent)` before KS-test.

3. **Seasonal false positives** — annual baseline inflated by winter months triggers false drift in April. Fix: seasonal baseline mode comparing same-month historical data.

**Code generated (fix):**
```python
def get_recent_window(city, days=7):
    # Aggregate to one reading per hour — matches pipeline.py aggregation
    combined["hour"] = combined["timestamp"].dt.floor("h")
    hourly = (
        combined
        .groupby("hour")["aqi"]
        .mean()
        .reset_index()
        .rename(columns={"hour": "timestamp"})
        .sort_values("timestamp")
    )
    return hourly["aqi"].dropna()

def check_drift(city, window_days=7, seasonal=False):
    # Subsample baseline to remove sample size bias
    if len(baseline) > len(recent):
        baseline = baseline.sample(n=len(recent), random_state=42)
    ks_stat, p_value = ks_2samp(baseline.values, recent.values)
```

---

### 2.4 FastAPI Inference Pipeline

**Query (paraphrased):** "write build_future_dataframe with fallback chain for regressors"

**AI Response:** Generated `build_future_dataframe` in `src/api/main.py` with a three-level fallback chain per regressor type: weather from OpenMeteo → recent historical mean; AQI lag from AQICN live → last known from disk → city recent mean. Also generated helper functions `_get_recent_weather_means`, `_get_recent_aqi_mean`, and `_get_last_known_aqi`.

---

### 2.5 MLflow Path Fix

**Query (paraphrased):** "models trained locally have host paths but FastAPI runs in Docker and cannot find them"

**AI Response:** Claude diagnosed that artifact paths in `mlflow.db` contained host machine paths (e.g., `/home/astitva/...`) instead of container paths (`/app/mlflow/...`). Provided SQLite UPDATE commands to fix all four affected tables:

```bash
sqlite3 mlflow/mlflow.db "
UPDATE logged_models
  SET artifact_location = REPLACE(artifact_location,
      rtrim(artifact_location, replace(artifact_location, '/app', '')), '/app')
  WHERE artifact_location NOT LIKE '/app/%';
"
```

---

### 2.6 AQICN Multi-Station Averaging

**Query:** "Delhi aqi is usually high around the day but aqicn on my dashboard shows relatively lower number. Similarly bangalore has lower no but my dashboard shows higher no. Just because it picks one sensor data that happens to be in polluted area. How to solve?"

**AI Response:** Identified that the AQICN named city slug returns one representative station, not a city average. Provided solution using AQICN search API to find all station UIDs per city, then fetching them in parallel using `ThreadPoolExecutor` and computing the median (not mean, to handle outlier stations like Bengaluru Silk Board with AQI=549).

**Code generated:**
```python
def fetch_current_aqi(city: str) -> dict | None:
    uids = CITY_STATION_UIDS.get(city, [])
    readings = []
    with ThreadPoolExecutor(max_workers=len(uids)) as executor:
        futures = {executor.submit(fetch_uid, uid): uid for uid in uids}
        for future in as_completed(futures):
            result = future.result()
            if result:
                readings.append(result)
    aqis       = sorted([r["aqi"] for r in readings])
    median_aqi = int(aqis[len(aqis) // 2])
    return {..., "aqi": median_aqi, "stations_used": len(readings)}
```

---

### 2.7 DVC Pipeline Setup

**Query (paraphrased):** "dvc dag ERROR: output 'data/baseline' is specified in both data/baseline.dvc and build_features stage"

**AI Response:** Explained that standalone `.dvc` pointer files conflict with `dvc.yaml` stage outputs when both track the same directory. Advised running `dvc remove data/baseline.dvc` (which removes the pointer but preserves cached data), then running `dvc commit` to lock the current state as the new baseline. Also clarified that `dvc remove` does not delete DVC cache or DagsHub remote data — all historical versions remain accessible via git tags.

---

### 2.8 SARIMA OOM and Removal

**Query (paraphrased):** "SARIMA is taking over 3 minutes and getting OOM killed even with 4GB"

**AI Response:** Explained that SARIMA(1,1,1)(1,1,1,24) on 32k rows peaks at 8-15GB RAM during Kalman filter fitting — not fixable with more memory. Recommended removing SARIMA from the drift-triggered retrain DAG since Prophet is the only served model; SARIMA is only useful as a one-time baseline comparison during initial training.

---

### 2.9 Prophet Training Inside Docker

**Query (paraphrased):** "If I run trainer.py locally will models be trained on local or container? SARIMA wasn't able to train."

**AI Response:** Explained that training inside the container writes artifacts to `/app/mlflow/` (bind mounted to `./mlflow/` on host), so files persist locally. Diagnosed SARIMA memory spike (4-6GB for SARIMA (1,1,1)(1,1,1,24) on 30,000 rows) as memory pressure issue. Provided fix to subsample SARIMA training data to last 2 years (17,520 rows) and temporarily add swap space.


---

### 2.10 Test Suite

**Query (paraphrased):** "Modify tests for the AQI prediction system to resolve errors"

**AI Response:** Reviewed the test file and fixed mock patching paths to match the actual import structure, updated fixture data to include all required Prophet regressor columns, and corrected integration test base URLs. Ensured unit tests run without a live server by properly isolating external dependencies via `unittest.mock.patch`. Final suite: 72 tests (64 unit + 8 integration), all passing.

---

### 2.11 Forecast Start Date Bug

**Query (paraphrased):** "Forecast is being done on 27 Nov 2025 instead of Apr 18 — why?"

**AI Response:** Diagnosed that Prophet's `make_future_dataframe` generates dates starting from the last date in training data (Nov 26, 2025), not from today. Fixed by building the `future` DataFrame directly from `datetime.utcnow()` in the FastAPI `/forecast` endpoint.

```python
now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
future = pd.DataFrame({
    "ds": pd.date_range(start=now, periods=hours, freq="H")
})
```

---

### 2.12 NaN in Prophet Regressor

**Query (paraphrased):** "ValueError: Found NaN in column 'temp_inversion'"

**AI Response:** Explained that Prophet rejects any NaN in regressors — the check `notna().sum() > 0` passes if even one row is non-null but the full column must be NaN-free. Fix: fill NaN with `0.0` for all regressors before `model.fit()`.

---

### 2.13 Airflow Parallelism and Memory Tuning

**Query (paraphrased):** "SARIMA is slow in container — should I increase memory or reduce pool to 1?"

**AI Response:** Explained that `AIRFLOW_PARALLELISM` controls global task concurrency across all DAGs, while `AIRFLOW_MAX_ACTIVE_TASKS_PER_DAG` limits concurrency within a single DAG. Recommended `RETRAIN_POOL_SLOTS=1` to serialize city retraining. Explained that `mem_limit` on the scheduler container caps all subprocesses including training — 4GB was insufficient for SARIMA peak memory of 8-15GB.

---

### 2.14 AlertManager Fixes

**Query (paraphrased):** "alertmanager is restarting — sed error and config load fails"

**AI Response:** Diagnosed two issues: (1) `body:` is not a valid field in Alertmanager email config — correct field is `text:`; (2) `sed -i` fails on Docker bind-mounted files because the file is read-only. Fix: copy config to `/tmp/` first, run `sed` on the copy. Also fixed `smtp_require_tls: false` → `true` for Mailtrap STARTTLS on port 2525, and added `--web.external-url=http://localhost:9093` so "View in Alertmanager" email links resolve correctly.

---

### 2.15 Miscellaneous Code and Fixes

Additional AI contributions:

- **IST timezone conversion fix** in `frontend/app.py` — `dt.replace(tzinfo=timezone.utc)` before `astimezone(IST)` to avoid timezone-naive datetime errors
- **Airflow 3.x compatibility** — `schedule=` instead of `schedule_interval=`, `pendulum.now().subtract()` instead of `days_ago()`, API v2 JWT endpoints
- **`compute_baseline_stats`** — monthly stratified baselines for seasonal drift detection
- **`dvc.yaml`** — complete pipeline file with correct stage dependencies after resolving overlapping output conflicts
- **`_parse_results` fix** in `openaq.py` — replacing `drop_duplicates` with `groupby.mean()` for intra-hour multi-reading aggregation
- **MLflow DB path fixes** — SQLite UPDATE commands to fix host paths (`/home/astitva/...`) to container paths (`/app/mlflow/...`) in experiments and runs tables
- **`reload-models` endpoint bug** — identified hyphen vs underscore (`/reload_models` → `/reload-models`) causing 404 in retrain DAG
- **Docker volume consistency** — diagnosed that `./mlflow` mounted at different paths across containers caused artifact writes to fail
- **FAB migration** — `AIRFLOW__CORE__AUTH_MANAGER`, `apache-airflow-providers-fab`, `airflow users create` replacing SimpleAuthManager
- **`aqi_retrain` skip logic** — city skip inside `retrain_city` when not in drifted list, single DAG run for all cities instead of one run per city


## 3. AI Use in Code Comments and Docstrings

All docstrings and inline comments throughout the codebase were generated with AI assistance. This includes:

- Module-level docstrings explaining file purpose, usage, and data flow (e.g., `openaq.py`, `drift.py`, `trainer.py`, `main.py`, etc.)
- Function docstrings explaining algorithm steps, parameter descriptions, return types, and fallback chains
- Inline comments explaining non-obvious logic (e.g., `# KS-test statistical power increases with N`, `# Prophet not thread-safe — workers=1`, `# sort_order=desc is ignored by OpenAQ /hours endpoint`)
- ASCII art flow diagrams inside DAG files showing task dependencies
- In generating template for UI in Streamlit
- For HLD, LLD diagram generation
- Section separator comments (`# ── Layer name ──────`)

## 4. Reflection

AI assistance  accelerated development of boilerplate-heavy components (Docker configs, Airflow DAGs, Prometheus configs, test suites) and was particularly valuable for diagnosing subtle statistical bugs like the KS-test sample size bias and the multi-station data aggregation mismatch.

The most valuable AI contributions were not code generation but explanations — understanding *why* `trigger_rule="all_done"` was needed, *why* bind mounts were better than volumes for this use case, and *why* the KS-test was producing p≈0. These explanations helped build genuine understanding of the system rather than just producing working code.


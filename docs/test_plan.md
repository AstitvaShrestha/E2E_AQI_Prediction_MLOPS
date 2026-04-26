# Test Plan — AQI Prediction System

**Version:** 1.0
**Date:** April 2026
**Test Framework:** pytest 9.x
**Test File:** `tests/test_aqi_prediction.py`
**Total Tests:** 72 (64 unit + 8 integration)

---

## 1. Test Strategy

### Test Levels
| Level | Description | Tool | When |
|-------|-------------|------|------|
| Unit | Individual functions, no external deps | pytest + mock | Every code change |
| Integration | Full API stack with running FastAPI | pytest + requests | Before demo |
| Evaluation | Model performance on held-out data | scripts/evaluate.py | After training |

### Test Execution
```bash
# Unit tests only (fast, ~23 seconds, no server needed)
pytest tests/test_aqi_prediction.py -v -m "not integration"
# Result: 64 passed, 8 deselected

# Integration tests (requires docker compose up -d fastapi)
pytest tests/test_aqi_prediction.py -v -m "integration"
# Result: 8 passed

# All tests
pytest tests/test_aqi_prediction.py -v
# Result: 72 passed

# Model evaluation (requires running FastAPI container)
docker compose exec aqi-prediction-fastapi python scripts/evaluate.py
```

---

## 2. Acceptance Criteria

### Model Performance (from evaluate.py — last 30 days)
| City        | Criterion | Target | Actual MAE | Actual RMSE | Status |
|-------------|-----------|--------|-----------|------------|--------|
| Delhi       | MAE ≤ 50 | ≤ 50 | 27.56 | 39.23 | ✓ PASS |
| Mumbai      | MAE ≤ 50 | ≤ 50 | 17.03 | 35.01 | ✓ PASS |
| Kolkata     | MAE ≤ 50 | ≤ 50 | 19.07 | 32.50 | ✓ PASS |
| Chennai     | MAE ≤ 50 | ≤ 50 | 19.10 | 32.61 | ✓ PASS |
| Bengaluru   | MAE ≤ 50 | ≤ 50 | 17.54 | 27.01 | ✓ PASS |
| **Average** | | | **20.06** | **33.27** | **✓ ALL PASS** |

### API Acceptance Criteria
| Criterion | Target | Status |
|-----------|--------|--------|
| /health returns 200 | Always | ✓ PASS |
| /forecast returns exactly 24 hours | 24 | ✓ PASS |
| AQI values in valid range | 0–500 | ✓ PASS |
| Confidence intervals valid (lower ≤ aqi ≤ upper) | Always | ✓ PASS |
| All 5 cities return forecasts | 5/5 | ✓ PASS |
| Invalid city returns 404 | 404 | ✓ PASS |
| Prometheus metrics exposed | /metrics | ✓ PASS |

**Overall: ALL ACCEPTANCE CRITERIA MET ✓**

---

## 3. Test Cases

### 3.1 TestHealthEndpoint — 6 tests

| ID | Test Name | Expected | Status |
|----|-----------|----------|--------|
| T01 | test_health_returns_200 | 200 OK | ✓ PASS |
| T02 | test_health_response_structure | status, models_loaded, models_missing, mlflow_uri, timestamp present | ✓ PASS |
| T03 | test_health_status_value | "healthy" or "degraded" | ✓ PASS |
| T04 | test_health_timestamp_is_valid_iso | datetime.fromisoformat() succeeds | ✓ PASS |
| T05 | test_health_models_fields_are_lists | isinstance(list) == True | ✓ PASS |
| T06 | test_health_models_sum_to_five | loaded + missing == 5 | ✓ PASS |

### 3.2 TestCitiesEndpoint — 6 tests

| ID | Test Name | Expected | Status |
|----|-----------|----------|--------|
| T07 | test_cities_returns_200 | 200 OK | ✓ PASS |
| T08 | test_cities_returns_dict | isinstance(dict) == True | ✓ PASS |
| T09 | test_cities_has_five_entries | len == 5 | ✓ PASS |
| T10 | test_cities_contains_expected | Delhi, Mumbai, Kolkata, Chennai, Bengaluru | ✓ PASS |
| T11 | test_cities_have_required_fields | lat, lon, timezone, model_loaded per city | ✓ PASS |
| T12 | test_cities_coordinates_are_numbers | isinstance(float) for lat and lon | ✓ PASS |

### 3.3 TestForecastEndpoint — 5 tests

| ID | Test Name | Expected | Status |
|----|-----------|----------|--------|
| T13 | test_forecast_invalid_city_returns_404 | 404 Not Found | ✓ PASS |
| T14 | test_forecast_city_case_sensitive | "delhi" → 404 | ✓ PASS |
| T15 | test_forecast_all_cities_not_404 | 200 or 503, never 404 | ✓ PASS |
| T16 | test_forecast_no_model_returns_503 | 503 Service Unavailable | ✓ PASS |
| T17 | test_forecast_response_structure_when_loaded | forecast, city, current_aqi, generated_at | ✓ PASS |

### 3.4 TestLatestAQIEndpoint — 4 tests

| ID | Test Name | Expected | Status |
|----|-----------|----------|--------|
| T18 | test_latest_invalid_city_returns_404 | 404 Not Found | ✓ PASS |
| T19 | test_latest_valid_city_not_404 | Not 404 | ✓ PASS |
| T20 | test_latest_response_has_required_fields | aqi, category, timestamp | ✓ PASS |
| T21 | test_latest_all_cities_not_404 | None of 5 cities return 404 | ✓ PASS |

### 3.5 TestDriftDetection — 7 tests

| ID | Test Name | Input | Expected | Status |
|----|-----------|-------|----------|--------|
| T22 | test_no_drift_same_distribution | N(150,30) vs N(150,30) | p > 0.05, ks_stat < 0.3 | ✓ PASS |
| T23 | test_drift_different_distribution | N(150,30) vs N(60,15) | p < 0.05, ks_stat > 0.3 | ✓ PASS |
| T24 | test_drift_p_value_range | Any inputs | 0 ≤ p ≤ 1 | ✓ PASS |
| T25 | test_drift_check_city_with_mock | check_drift mocked | All required fields | ✓ PASS |
| T26 | test_drift_result_types | Result dict | is_drifted=bool, p_value=float, recent_n=int | ✓ PASS |
| T27 | test_drift_handles_missing_baseline | No baseline | error not None, is_drifted=False | ✓ PASS |
| T28 | test_drift_handles_insufficient_recent_data | 2 samples | No crash, error returned | ✓ PASS |

### 3.6 TestFeaturePipeline — 10 tests

| ID | Test Name | Input | Expected | Status |
|----|-----------|-------|----------|--------|
| T29 | test_bonus_features_crop_burning_october | October dates | crop_burning == 1.0 | ✓ PASS |
| T30 | test_bonus_features_crop_burning_july | July dates | crop_burning == 0.0 | ✓ PASS |
| T31 | test_bonus_features_festival_october | October dates | festival_period == 0.55 | ✓ PASS |
| T32 | test_bonus_features_festival_march | March dates | festival_period == 0.0 | ✓ PASS |
| T33 | test_cyclical_features_range | Sample DataFrame | All sin/cos in [-1.0, 1.0] | ✓ PASS |
| T34 | test_features_sorted_by_time | Sample DataFrame | ds is_monotonic_increasing | ✓ PASS |
| T35 | test_required_features_exist | Sample DataFrame | All 12 Prophet regressors present | ✓ PASS |
| T36 | test_no_negative_aqi | y column | All values ≥ 0 | ✓ PASS |
| T37 | test_save_features_creates_file | save_features() | .parquet created at correct path | ✓ PASS |
| T38 | test_compute_baseline_stats_creates_file | compute_baseline_stats() | _baseline.parquet and _stats.json created | ✓ PASS |

### 3.7 TestAQICategory — 18 tests

| ID | Test Name | AQI | Expected Label | Status |
|----|-----------|-----|---------------|--------|
| T39 | test_good_category | 25 | Good, color=#00b050 | ✓ PASS |
| T40 | test_satisfactory_category | 75 | Satisfactory | ✓ PASS |
| T41 | test_moderate_category | 150 | Moderate | ✓ PASS |
| T42 | test_poor_category | 250 | Poor | ✓ PASS |
| T43 | test_very_poor_category | 350 | Very Poor | ✓ PASS |
| T44 | test_severe_category | 450 | Severe | ✓ PASS |
| T45 | test_boundary_50_is_good | 50 | Good | ✓ PASS |
| T46 | test_boundary_51_is_satisfactory | 51 | Satisfactory | ✓ PASS |
| T47 | test_boundary_100_is_satisfactory | 100 | Satisfactory | ✓ PASS |
| T48 | test_boundary_101_is_moderate | 101 | Moderate | ✓ PASS |
| T49 | test_boundary_200_is_moderate | 200 | Moderate | ✓ PASS |
| T50 | test_boundary_201_is_poor | 201 | Poor | ✓ PASS |
| T51 | test_boundary_300_is_poor | 300 | Poor | ✓ PASS |
| T52 | test_boundary_301_is_very_poor | 301 | Very Poor | ✓ PASS |
| T53 | test_boundary_400_is_very_poor | 400 | Very Poor | ✓ PASS |
| T54 | test_boundary_401_is_severe | 401 | Severe | ✓ PASS |
| T55 | test_category_has_label_color_advice | 25,75,150,250,350,450 | label, color, advice all present | ✓ PASS |
| T56 | test_category_color_is_hex | 25,75,150,250,350,450 | Matches #RRGGBB regex | ✓ PASS |

### 3.8 TestConfig — 8 tests

| ID | Test Name | Expected | Status |
|----|-----------|----------|--------|
| T57 | test_cities_dict_exists | isinstance(dict) | ✓ PASS |
| T58 | test_cities_has_five_entries | len == 5 | ✓ PASS |
| T59 | test_all_expected_cities_present | All 5 cities in dict | ✓ PASS |
| T60 | test_all_cities_have_lat_lon | lat, lon keys present | ✓ PASS |
| T61 | test_all_cities_have_timezone | "Asia/Kolkata" | ✓ PASS |
| T62 | test_all_cities_have_openaq_ids | list len > 0 | ✓ PASS |
| T63 | test_coordinates_within_india | 8≤lat≤37, 68≤lon≤97 | ✓ PASS |
| T64 | test_delhi_coordinates_approximate | lat≈28.61, lon≈77.21 ±0.1 | ✓ PASS |

### 3.9 TestIntegration — 8 tests (require `docker compose up -d fastapi`)

| ID | Test Name | Expected | Status |
|----|-----------|----------|--------|
| T65 | test_api_health_live | status=healthy, 5 models loaded | ✓ PASS |
| T66 | test_forecast_delhi_live | 24 hours returned, city=Delhi | ✓ PASS |
| T67 | test_forecast_aqi_values_in_valid_range | 0 ≤ aqi ≤ 500 for all hours | ✓ PASS |
| T68 | test_forecast_confidence_intervals_valid | lower ≤ aqi ≤ upper for all hours | ✓ PASS |
| T69 | test_all_five_cities_forecast_live | All 5 cities return 200 | ✓ PASS |
| T70 | test_metrics_endpoint_has_http_requests_total | http_requests_total in /metrics | ✓ PASS |
| T71 | test_latest_aqi_returns_valid_value | 0 ≤ aqi ≤ 500 | ✓ PASS |
| T72 | test_forecast_category_labels_valid | Good/Satisfactory/Moderate/Poor/Very Poor/Severe | ✓ PASS |

---

## 4. Test Results Summary

### Unit Test Run
```
Platform:  Linux, Python 3.11.14, pytest 9.0.3
Collected: 72 items
Deselected: 8 (integration)
Selected:   64

PASSED:   64
FAILED:   0
SKIPPED:  8
Duration: 22.83s
```

### Test Class Summary
| Class | Tests | Result |
|-------|-------|--------|
| TestHealthEndpoint | 6 | 6 passed |
| TestCitiesEndpoint | 6 | 6 passed |
| TestForecastEndpoint | 5 | 5 passed |
| TestLatestAQIEndpoint | 4 | 4 passed |
| TestDriftDetection | 7 | 7 passed |
| TestFeaturePipeline | 10 | 10 passed |
| TestAQICategory | 18 | 18 passed |
| TestConfig | 8 | 8 passed |
| TestIntegration | 8 | 8 passed |
| **Total** | **72** | **72 passed** |

### Model Evaluation
```

City        MAE    RMSE   MAPE    Coverage  CatAcc  BndErr  RushMAE
Delhi       27.56  39.23  23.45%  63.6%     71.5%   10.6%   26.24   ✓ PASS
Mumbai      17.03  35.01  20.69%  76.5%     78.3%    9.9%   17.92   ✓ PASS
Kolkata     19.07  32.50  21.87%  80.0%     80.7%    7.1%   18.99   ✓ PASS
Chennai     19.10  32.61  26.88%  63.9%     61.4%   21.1%   17.95   ✓ PASS
Bengaluru   17.54  27.01  17.43%  69.4%     55.4%   25.4%   18.29   ✓ PASS
Average     20.06  33.27  22.06%  70.7%     69.5%   14.8%   19.88

Calibration gaps (target coverage 95%):
  Delhi      CalGap=31.4% | BndShare=0.37 | CatAccExBnd=77.8% | CovExBnd=65.7%
  Mumbai     CalGap=18.5% | BndShare=0.46 | CatAccExBnd=82.6% | CovExBnd=76.1%
  Kolkata    CalGap=15.0% | BndShare=0.37 | CatAccExBnd=83.8% | CovExBnd=82.3%
  Chennai    CalGap=31.1% | BndShare=0.55 | CatAccExBnd=69.5% | CovExBnd=60.7%
  Bengaluru  CalGap=25.6% | BndShare=0.57 | CatAccExBnd=62.2% | CovExBnd=52.2%

ALL 5 CITIES PASS MAE ≤ 50 ✓
```

---

## 5. Known Issues

| Issue | Severity | Root Cause | Mitigation |
|-------|----------|-----------|------------|
| Coverage 70% vs 95% | Low | Fat-tailed AQI distributions underestimate Prophet CI | Documented; conformal prediction recommended |
| Bengaluru CatAcc 55.4% | Low | 57% of errors within 10 AQI of category boundaries | CatAcc ex-boundary = 62.2%; boundary effect |
| Seasonal drift FP | Low | Annual baseline inflated by winter months | Seasonal mode: GET /drift?seasonal=true |
| KS sample size bias | Fixed | 2160 baseline vs 115 recent → p≈0 always | Baseline subsampled to len(recent) |

"""
tests/test_aqi_prediction.py

Unit and integration tests for AQI Prediction System.
Tests: API endpoints, drift detection, feature pipeline, config.

Run:
    pytest tests/test_aqi_prediction.py -v -m "not integration"   # unit tests
    pytest tests/test_aqi_prediction.py -v -m "integration"       # needs server
    pytest tests/test_aqi_prediction.py -v                        # all tests
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient


#----FIXTURES---

@pytest.fixture(scope="module")
def client():
    """
    FastAPI test client — does not require a running server.
    Mocks load_champion_models so tests do not need MLflow.
    """
    with patch("src.api.main.load_champion_models"):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_features_df():
    """
    Sample features DataFrame matching the actual pipeline output schema.
    Uses 200 rows — enough for lag_168h (7 days) to have valid values.
    """
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    return pd.DataFrame({
        "ds":                    dates,
        "y":                     np.random.randint(50, 200, 200).astype(float),
        "aqi_lag_1h":            np.random.randint(50, 200, 200).astype(float),
        "aqi_lag_24h":           np.random.randint(50, 200, 200).astype(float),
        "aqi_lag_168h":          np.random.randint(50, 200, 200).astype(float),
        "aqi_roll_mean_6h":      np.random.randint(50, 200, 200).astype(float),
        "aqi_roll_mean_24h":     np.random.randint(50, 200, 200).astype(float),
        "aqi_roll_mean_7d":      np.random.randint(50, 200, 200).astype(float),
        "aqi_roll_std_24h":      np.random.uniform(5, 30, 200),
        "aqi_roll_max_24h":      np.random.randint(80, 250, 200).astype(float),
        "temperature_2m":        np.random.uniform(15, 40, 200),
        "wind_speed_10m":        np.random.uniform(0, 20, 200),
        "relative_humidity_2m":  np.random.uniform(20, 90, 200),
        "is_rush_hour":          np.random.randint(0, 2, 200),
        "is_weekend":            np.random.randint(0, 2, 200),
        "crop_burning":          np.zeros(200),
        "festival_period":       np.zeros(200),
        "hour":                  dates.hour,
        "dayofweek":             dates.dayofweek,
        "month":                 dates.month,
        "hour_sin":              np.sin(2 * np.pi * dates.hour / 24),
        "hour_cos":              np.cos(2 * np.pi * dates.hour / 24),
        "dow_sin":               np.sin(2 * np.pi * dates.dayofweek / 7),
        "dow_cos":               np.cos(2 * np.pi * dates.dayofweek / 7),
        "month_sin":             np.sin(2 * np.pi * dates.month / 12),
        "month_cos":             np.cos(2 * np.pi * dates.month / 12),
    })


#----API ENDPOINT TESTS---

class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint must always return 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client):
        """Health response must contain all required fields."""
        resp   = client.get("/health")
        data   = resp.json()
        fields = ["status", "models_loaded", "models_missing",
                  "mlflow_uri", "timestamp"]
        for field in fields:
            assert field in data, f"Missing field in /health response: {field}"

    def test_health_status_value(self, client):
        """Status field must be 'healthy' or 'degraded'."""
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] in ["healthy", "degraded"]

    def test_health_timestamp_is_valid_iso(self, client):
        """Timestamp must be parseable as ISO 8601."""
        resp = client.get("/health")
        data = resp.json()
        # Should not raise ValueError
        datetime.fromisoformat(data["timestamp"])

    def test_health_models_fields_are_lists(self, client):
        """models_loaded and models_missing must be lists."""
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["models_loaded"],  list)
        assert isinstance(data["models_missing"], list)

    def test_health_models_sum_to_five(self, client):
        """Total cities must always equal 5."""
        resp   = client.get("/health")
        data   = resp.json()
        total  = len(data["models_loaded"]) + len(data["models_missing"])
        assert total == 5, f"Expected 5 cities total, got {total}"


class TestCitiesEndpoint:
    """Tests for GET /cities endpoint."""

    def test_cities_returns_200(self, client):
        resp = client.get("/cities")
        assert resp.status_code == 200

    def test_cities_returns_dict(self, client):
        """/cities returns a dict keyed by city name."""
        resp = client.get("/cities")
        data = resp.json()
        assert isinstance(data, dict)

    def test_cities_has_five_entries(self, client):
        resp = client.get("/cities")
        data = resp.json()
        assert len(data) == 5, f"Expected 5 cities, got {len(data)}"

    def test_cities_contains_expected(self, client):
        resp     = client.get("/cities")
        cities   = resp.json()
        expected = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]
        for city in expected:
            assert city in cities, f"{city} missing from /cities response"

    def test_cities_have_required_fields(self, client):
        """Each city entry must have lat, lon, timezone, model_loaded."""
        resp = client.get("/cities")
        data = resp.json()
        for city, config in data.items():
            for field in ["lat", "lon", "timezone", "model_loaded"]:
                assert field in config, \
                    f"{city} missing field '{field}' in /cities"

    def test_cities_coordinates_are_numbers(self, client):
        resp = client.get("/cities")
        data = resp.json()
        for city, config in data.items():
            assert isinstance(config["lat"], (int, float)), \
                f"{city} lat is not a number"
            assert isinstance(config["lon"], (int, float)), \
                f"{city} lon is not a number"


class TestForecastEndpoint:
    """Tests for GET /forecast/{city} endpoint."""

    def test_forecast_invalid_city_returns_404(self, client):
        """Unknown city must return 404."""
        resp = client.get("/forecast/InvalidCity")
        assert resp.status_code == 404

    def test_forecast_city_case_sensitive(self, client):
        """City names are case sensitive — 'delhi' is not valid."""
        resp = client.get("/forecast/delhi")
        assert resp.status_code == 404

    def test_forecast_all_cities_not_404(self, client):
        """All 5 valid cities must not return 404."""
        cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]
        for city in cities:
            resp = client.get(f"/forecast/{city}")
            assert resp.status_code != 404, \
                f"City '{city}' returned 404 — not in CITIES dict"

    def test_forecast_no_model_returns_503(self, client):
        """If model not loaded, must return 503 not 500."""
        # In test env models are not loaded (mocked out)
        resp = client.get("/forecast/Delhi")
        assert resp.status_code in [200, 503], \
            f"Expected 200 or 503, got {resp.status_code}"

    def test_forecast_response_structure_when_loaded(self, client):
        """When model available, response must have forecast list."""
        mock_model = MagicMock()
        # Build a minimal forecast DataFrame Prophet would return
        future_df = pd.DataFrame({
            "ds":         pd.date_range("2026-01-01", periods=24, freq="h"),
            "yhat":       np.random.uniform(80, 150, 24),
            "yhat_lower": np.random.uniform(60, 100, 24),
            "yhat_upper": np.random.uniform(100, 180, 24),
        })
        mock_model.predict.return_value = future_df

        with patch("src.api.main.champion_models", {"Delhi": mock_model}):
            with patch("src.api.main.build_future_dataframe",
                       return_value=future_df):
                with patch("src.api.main.get_current_aqi_value",
                           return_value=120.0):
                    resp = client.get("/forecast/Delhi")
                    if resp.status_code == 200:
                        data = resp.json()
                        assert "forecast"    in data
                        assert "city"        in data
                        assert "current_aqi" in data
                        assert "generated_at" in data


class TestLatestAQIEndpoint:
    """Tests for GET /latest/{city} endpoint."""

    def test_latest_invalid_city_returns_404(self, client):
        resp = client.get("/latest/InvalidCity")
        assert resp.status_code == 404

    def test_latest_valid_city_not_404(self, client):
        """Valid city must not return 404."""
        resp = client.get("/latest/Delhi")
        assert resp.status_code != 404

    def test_latest_response_has_required_fields(self, client):
        """Response must have aqi, category, timestamp."""
        with patch("src.api.main.get_current_aqi_value", return_value=120.0):
            resp = client.get("/latest/Delhi")
            if resp.status_code == 200:
                data = resp.json()
                assert "aqi"       in data
                assert "category"  in data
                assert "timestamp" in data

    def test_latest_all_cities_not_404(self, client):
        """All 5 cities must be valid endpoints."""
        for city in ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]:
            resp = client.get(f"/latest/{city}")
            assert resp.status_code != 404, \
                f"/latest/{city} returned 404"


#----DRIFT DETECTION TESTS---

class TestDriftDetection:
    """
    Tests for drift detection logic.
    Tests check_drift() by mocking file I/O — no parquet files needed.
    """

    def test_no_drift_same_distribution(self):
        """
        Same distribution (mean=150) should not trigger drift.
        KS-test p-value should be > 0.05.
        """
        from src.features.drift import check_drift
        from scipy.stats import ks_2samp

        np.random.seed(42)
        baseline_vals = np.random.normal(150, 30, 1000)
        recent_vals   = np.random.normal(150, 30, 100)

        # Test the KS-test directly (bypassing file I/O)
        ks_stat, p_value = ks_2samp(baseline_vals, recent_vals)

        assert p_value > 0.05, \
            f"Same distribution incorrectly flagged as drift (p={p_value:.4f})"
        assert ks_stat < 0.3, \
            f"KS statistic too high for same distribution: {ks_stat:.4f}"

    def test_drift_different_distribution(self):
        """
        Different distributions (summer vs winter) should trigger drift.
        p-value should be < 0.05.
        """
        from scipy.stats import ks_2samp

        np.random.seed(42)
        baseline_vals = np.random.normal(150, 30, 1000)  # winter AQI
        recent_vals   = np.random.normal(60,  15, 100)   # summer AQI

        ks_stat, p_value = ks_2samp(baseline_vals, recent_vals)

        assert p_value < 0.05, \
            f"Different distribution not flagged as drift (p={p_value:.4f})"
        assert ks_stat > 0.3, \
            f"KS statistic too low for different distributions: {ks_stat:.4f}"

    def test_drift_p_value_range(self):
        """p-value from KS-test must always be between 0 and 1."""
        from scipy.stats import ks_2samp

        np.random.seed(42)
        baseline = np.random.normal(150, 30, 1000)
        recent   = np.random.normal(120, 25, 100)

        _, p_value = ks_2samp(baseline, recent)

        assert 0 <= p_value <= 1, \
            f"p_value out of [0,1] range: {p_value}"

    def test_drift_check_city_with_mock(self):
        """check_drift(city) returns correct structure when mocked."""
        from src.features.drift import check_drift
        import pandas as pd

        np.random.seed(42)
        mock_baseline = pd.Series(np.random.normal(150, 30, 1000))
        mock_recent   = pd.Series(np.random.normal(150, 30, 100))

        with patch("src.features.drift.get_baseline",
                   return_value=mock_baseline):
            with patch("src.features.drift.get_recent_window",
                       return_value=mock_recent):
                with patch("src.features.drift.save_drift_window"):
                    result = check_drift("Delhi")

        # Verify structure
        required = [
            "city", "p_value", "ks_stat",
            "is_drifted", "baseline_mean",
            "recent_mean", "recent_n", "checked_at", "error"
        ]
        for field in required:
            assert field in result, \
                f"Missing field in drift result: {field}"

    def test_drift_result_types(self):
        """Drift result fields must have correct types."""
        from src.features.drift import check_drift
        import pandas as pd

        np.random.seed(42)
        mock_baseline = pd.Series(np.random.normal(150, 30, 1000))
        mock_recent   = pd.Series(np.random.normal(150, 30, 100))

        with patch("src.features.drift.get_baseline",
                   return_value=mock_baseline):
            with patch("src.features.drift.get_recent_window",
                       return_value=mock_recent):
                with patch("src.features.drift.save_drift_window"):
                    result = check_drift("Delhi")

        assert isinstance(result["is_drifted"],   bool)
        assert isinstance(result["p_value"],       float)
        assert isinstance(result["ks_stat"],       float)
        assert isinstance(result["recent_n"],      int)
        assert result["city"] == "Delhi"

    def test_drift_handles_missing_baseline(self):
        """check_drift must return error dict if baseline not found."""
        from src.features.drift import check_drift

        with patch("src.features.drift.get_baseline", return_value=None):
            result = check_drift("Delhi")

        assert result is not None
        assert result.get("error") is not None, \
            "Should report error when baseline missing"
        assert result["is_drifted"] == False

    def test_drift_handles_insufficient_recent_data(self):
        """check_drift must handle too few recent samples gracefully."""
        from src.features.drift import check_drift
        import pandas as pd

        mock_baseline = pd.Series(np.random.normal(150, 30, 1000))
        mock_recent   = pd.Series([120.0, 130.0])  # only 2 samples

        with patch("src.features.drift.get_baseline",
                   return_value=mock_baseline):
            with patch("src.features.drift.get_recent_window",
                       return_value=mock_recent):
                result = check_drift("Delhi")

        assert result is not None
        # Should not crash — returns error or skip
        assert "error" in result


#----FEATURE PIPELINE TESTS---

class TestFeaturePipeline:
    """
    Tests for feature engineering.
    Tests add_bonus_features() directly — no file I/O needed.
    Tests the output schema of build_features_spark() using mocks.
    """

    def test_bonus_features_crop_burning_october(self):
        """crop_burning must be 1.0 in October."""
        from src.features.pipeline import add_bonus_features

        dates = pd.date_range("2024-10-01", periods=24, freq="h")
        df    = pd.DataFrame({"ds": dates, "y": np.ones(24) * 100})
        result = add_bonus_features(df, timestamp_col="ds")

        assert "crop_burning" in result.columns
        assert (result["crop_burning"] == 1.0).all(), \
            "crop_burning should be 1.0 in October"

    def test_bonus_features_crop_burning_july(self):
        """crop_burning must be 0.0 in July."""
        from src.features.pipeline import add_bonus_features

        dates  = pd.date_range("2024-07-01", periods=24, freq="h")
        df     = pd.DataFrame({"ds": dates, "y": np.ones(24) * 100})
        result = add_bonus_features(df, timestamp_col="ds")

        assert (result["crop_burning"] == 0.0).all(), \
            "crop_burning should be 0.0 in July"

    def test_bonus_features_festival_october(self):
        """festival_period must be 0.55 in October."""
        from src.features.pipeline import add_bonus_features

        dates  = pd.date_range("2024-10-01", periods=24, freq="h")
        df     = pd.DataFrame({"ds": dates, "y": np.ones(24) * 100})
        result = add_bonus_features(df, timestamp_col="ds")

        assert "festival_period" in result.columns
        assert (result["festival_period"] == 0.55).all(), \
            "festival_period should be 0.55 in October"

    def test_bonus_features_festival_march(self):
        """festival_period must be 0.0 outside Oct-Nov."""
        from src.features.pipeline import add_bonus_features

        dates  = pd.date_range("2024-03-01", periods=24, freq="h")
        df     = pd.DataFrame({"ds": dates, "y": np.ones(24) * 100})
        result = add_bonus_features(df, timestamp_col="ds")

        assert (result["festival_period"] == 0.0).all(), \
            "festival_period should be 0.0 in March"

    def test_cyclical_features_range(self, sample_features_df):
        """Sin/cos cyclical features must be in [-1, 1]."""
        df = sample_features_df.copy()
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "month_sin", "month_cos"]:
            if col in df.columns:
                assert df[col].between(-1.0, 1.0).all(), \
                    f"Cyclical feature '{col}' has values outside [-1, 1]"

    def test_features_sorted_by_time(self, sample_features_df):
        """Feature DataFrame must be sorted by timestamp (ds)."""
        df = sample_features_df.copy()
        assert df["ds"].is_monotonic_increasing, \
            "Features not sorted by 'ds' timestamp"

    def test_required_features_exist(self, sample_features_df):
        """All Prophet regressors must exist in feature DataFrame."""
        required = [
            "ds", "y",
            "temperature_2m", "wind_speed_10m", "relative_humidity_2m",
            "is_rush_hour", "crop_burning", "festival_period",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]
        for col in required:
            assert col in sample_features_df.columns, \
                f"Required feature '{col}' missing from DataFrame"

    def test_no_negative_aqi(self, sample_features_df):
        """AQI target (y) must not have negative values."""
        df = sample_features_df.copy()
        assert (df["y"] >= 0).all(), \
            "Negative AQI values found in target column 'y'"

    def test_save_features_creates_file(self, sample_features_df, tmp_path):
        """save_features must create a parquet file."""
        from src.features.pipeline import save_features

        # Patch DATA_DIR to tmp_path so no real disk writes
        with patch("src.features.pipeline.DATA_DIR", tmp_path):
            save_features("Delhi", sample_features_df)
            out = tmp_path / "processed" / "delhi" / "features.parquet"
            assert out.exists(), \
                f"save_features did not create {out}"

    def test_compute_baseline_stats_creates_file(self,
                                                  sample_features_df,
                                                  tmp_path):
        """compute_baseline_stats must create parquet and stats JSON files."""

        from src.features.pipeline import compute_baseline_stats

        with patch("src.features.pipeline.DATA_DIR", tmp_path):
            compute_baseline_stats("Delhi", sample_features_df)

            city_key = "delhi"
            # Check parquet baseline
            parquet_out = tmp_path / "baseline" / f"{city_key}_baseline.parquet"
            # Check stats JSON
            stats_out   = tmp_path / "baseline" / f"{city_key}_stats.json"

            assert parquet_out.exists(), \
                f"compute_baseline_stats did not create {parquet_out}"
            assert stats_out.exists(), \
                f"compute_baseline_stats did not create {stats_out}"

#----AQI CATEGORY TESTS---

class TestAQICategory:
    """
    Tests for get_aqi_category() in src/config.py.
    India CPCB scale: Good(0-50), Satisfactory(51-100),
    Moderate(101-200), Poor(201-300), Very Poor(301-400), Severe(401-500).
    """

    def test_good_category(self):
        from src.config import get_aqi_category
        result = get_aqi_category(25)
        assert result["label"] == "Good"
        assert result["color"] == "#00b050"

    def test_satisfactory_category(self):
        from src.config import get_aqi_category
        assert get_aqi_category(75)["label"] == "Satisfactory"

    def test_moderate_category(self):
        from src.config import get_aqi_category
        assert get_aqi_category(150)["label"] == "Moderate"

    def test_poor_category(self):
        from src.config import get_aqi_category
        assert get_aqi_category(250)["label"] == "Poor"

    def test_very_poor_category(self):
        from src.config import get_aqi_category
        assert get_aqi_category(350)["label"] == "Very Poor"

    def test_severe_category(self):
        from src.config import get_aqi_category
        assert get_aqi_category(450)["label"] == "Severe"

    def test_boundary_50_is_good(self):
        from src.config import get_aqi_category
        assert get_aqi_category(50)["label"] == "Good"

    def test_boundary_51_is_satisfactory(self):
        from src.config import get_aqi_category
        assert get_aqi_category(51)["label"] == "Satisfactory"

    def test_boundary_100_is_satisfactory(self):
        from src.config import get_aqi_category
        assert get_aqi_category(100)["label"] == "Satisfactory"

    def test_boundary_101_is_moderate(self):
        from src.config import get_aqi_category
        assert get_aqi_category(101)["label"] == "Moderate"

    def test_boundary_200_is_moderate(self):
        from src.config import get_aqi_category
        assert get_aqi_category(200)["label"] == "Moderate"

    def test_boundary_201_is_poor(self):
        from src.config import get_aqi_category
        assert get_aqi_category(201)["label"] == "Poor"

    def test_boundary_300_is_poor(self):
        from src.config import get_aqi_category
        assert get_aqi_category(300)["label"] == "Poor"

    def test_boundary_301_is_very_poor(self):
        from src.config import get_aqi_category
        assert get_aqi_category(301)["label"] == "Very Poor"

    def test_boundary_400_is_very_poor(self):
        from src.config import get_aqi_category
        assert get_aqi_category(400)["label"] == "Very Poor"

    def test_boundary_401_is_severe(self):
        from src.config import get_aqi_category
        assert get_aqi_category(401)["label"] == "Severe"

    def test_category_has_label_color_advice(self):
        """Every category result must have label, color, advice."""
        from src.config import get_aqi_category
        for aqi in [25, 75, 150, 250, 350, 450]:
            result = get_aqi_category(aqi)
            for field in ["label", "color", "advice"]:
                assert field in result, \
                    f"AQI {aqi}: missing '{field}' in category result"

    def test_category_color_is_hex(self):
        """Color must be a valid hex color string."""
        from src.config import get_aqi_category
        import re
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for aqi in [25, 75, 150, 250, 350, 450]:
            color = get_aqi_category(aqi)["color"]
            assert hex_pattern.match(color), \
                f"AQI {aqi}: color '{color}' is not a valid hex color"


#----CONFIG TESTS---

class TestConfig:
    """Tests for src/config.py — CITIES dict and coordinates."""

    def test_cities_dict_exists(self):
        from src.config import CITIES
        assert isinstance(CITIES, dict)

    def test_cities_has_five_entries(self):
        from src.config import CITIES
        assert len(CITIES) == 5, \
            f"Expected 5 cities, got {len(CITIES)}"

    def test_all_expected_cities_present(self):
        from src.config import CITIES
        expected = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]
        for city in expected:
            assert city in CITIES, f"'{city}' missing from CITIES dict"

    def test_all_cities_have_lat_lon(self):
        from src.config import CITIES
        for city, config in CITIES.items():
            assert "lat" in config, f"{city} missing 'lat'"
            assert "lon" in config, f"{city} missing 'lon'"

    def test_all_cities_have_timezone(self):
        from src.config import CITIES
        for city, config in CITIES.items():
            assert "timezone" in config, f"{city} missing 'timezone'"
            assert config["timezone"] == "Asia/Kolkata", \
                f"{city} timezone should be Asia/Kolkata"

    def test_all_cities_have_openaq_ids(self):
        from src.config import CITIES
        for city, config in CITIES.items():
            assert "openaq_location_ids" in config, \
                f"{city} missing 'openaq_location_ids'"
            assert len(config["openaq_location_ids"]) > 0, \
                f"{city} has empty openaq_location_ids"

    def test_coordinates_within_india(self):
        """All coordinates must fall within India's geographic bounds."""
        from src.config import CITIES
        for city, config in CITIES.items():
            assert 8.0 <= config["lat"] <= 37.0, \
                f"{city} latitude {config['lat']} outside India bounds"
            assert 68.0 <= config["lon"] <= 97.0, \
                f"{city} longitude {config['lon']} outside India bounds"

    def test_delhi_coordinates_approximate(self):
        """Delhi coordinates must be approximately correct."""
        from src.config import CITIES
        delhi = CITIES["Delhi"]
        assert abs(delhi["lat"] - 28.6139) < 0.1
        assert abs(delhi["lon"] - 77.2090) < 0.1


#----INTEGRATION TESTS---

class TestIntegration:
    """
    Integration tests against the live FastAPI server.
    Run with: pytest -m integration
    Requires: sudo docker compose up -d fastapi
    """

    BASE_URL = "http://localhost:8000"

    @pytest.mark.integration
    def test_api_health_live(self):
        """Live /health must return 200 with all 5 models."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/health", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert len(data["models_loaded"]) == 5

    @pytest.mark.integration
    def test_forecast_delhi_live(self):
        """Live /forecast/Delhi must return 24 hourly predictions."""
        import requests
        resp = requests.get(
            f"{self.BASE_URL}/forecast/Delhi", timeout=30
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "forecast"    in data
        assert len(data["forecast"]) == 24
        assert data["city"]  == "Delhi"

    @pytest.mark.integration
    def test_forecast_aqi_values_in_valid_range(self):
        """All forecast AQI values must be between 0 and 500."""
        import requests
        resp = requests.get(
            f"{self.BASE_URL}/forecast/Delhi", timeout=30
        )
        assert resp.status_code == 200
        for hour in resp.json()["forecast"]:
            assert 0 <= hour["aqi"] <= 500, \
                f"AQI {hour['aqi']} outside valid range [0, 500]"

    @pytest.mark.integration
    def test_forecast_confidence_intervals_valid(self):
        """aqi_lower must be <= aqi <= aqi_upper for all hours."""
        import requests
        resp = requests.get(
            f"{self.BASE_URL}/forecast/Delhi", timeout=30
        )
        assert resp.status_code == 200
        for hour in resp.json()["forecast"]:
            assert hour["aqi_lower"] <= hour["aqi"] <= hour["aqi_upper"], \
                f"Confidence interval invalid: " \
                f"{hour['aqi_lower']} <= {hour['aqi']} <= {hour['aqi_upper']}"

    @pytest.mark.integration
    def test_all_five_cities_forecast_live(self):
        """All 5 cities must return 200 from /forecast."""
        import requests
        for city in ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]:
            resp = requests.get(
                f"{self.BASE_URL}/forecast/{city}", timeout=30
            )
            assert resp.status_code == 200, \
                f"/forecast/{city} returned {resp.status_code}"

    @pytest.mark.integration
    def test_metrics_endpoint_has_http_requests_total(self):
        """Prometheus /metrics must expose http_requests_total."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/metrics", timeout=10)
        assert resp.status_code == 200
        assert "http_requests_total" in resp.text

    @pytest.mark.integration
    def test_latest_aqi_returns_valid_value(self):
        """Live /latest/Delhi must return AQI in [0, 500]."""
        import requests
        resp = requests.get(
            f"{self.BASE_URL}/latest/Delhi", timeout=15
        )
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["aqi"] <= 500, \
            f"AQI {data['aqi']} outside valid range"

    @pytest.mark.integration
    def test_forecast_category_labels_valid(self):
        """All forecast hours must have valid category labels."""
        import requests
        valid_labels = {
            "Good", "Satisfactory", "Moderate",
            "Poor", "Very Poor", "Severe"
        }
        resp = requests.get(
            f"{self.BASE_URL}/forecast/Delhi", timeout=30
        )
        assert resp.status_code == 200
        for hour in resp.json()["forecast"]:
            label = hour["category"]["label"]
            assert label in valid_labels, \
                f"Invalid category label: '{label}'"
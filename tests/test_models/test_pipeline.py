"""
Tests for Model Training Pipeline.

Tests the pipeline construction and feature extraction logic.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from src.models.pipeline import FraudFeatureExtractor, create_fraud_pipeline


class TestFraudFeatureExtractor:
    """Test suite for custom feature extractor."""

    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        extractor = FraudFeatureExtractor()

        # Test data: NYC to LA (approx 3944 km)
        data = pd.DataFrame(
            {
                "lat": [40.7128],
                "long": [-74.0060],
                "merch_lat": [34.0522],
                "merch_long": [-118.2437],
            }
        )

        result = extractor.transform(data)

        assert "distance_km" in result.columns
        # Rough check (actual is ~3944 km)
        assert 3900 < result["distance_km"].iloc[0] < 4000

    def test_cyclical_time_features(self):
        """Test cyclical encoding of hour and day."""
        extractor = FraudFeatureExtractor()

        data = pd.DataFrame(
            {
                "trans_date_trans_time": ["2019-01-01 12:00:00"]  # Noon on Tuesday
            }
        )

        result = extractor.transform(data)

        # Check features exist
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns

        # Noon (12) should be at pi (sin≈0, cos≈-1)
        assert abs(result["hour_sin"].iloc[0]) < 0.1
        assert result["hour_cos"].iloc[0] < 0

    def test_amount_log_transform(self):
        """Test log transformation of amount."""
        extractor = FraudFeatureExtractor()

        data = pd.DataFrame({"amt": [100.0, 1000.0]})

        result = extractor.transform(data)

        assert "amt_log" in result.columns
        # log1p(100) ≈ 4.615, log1p(1000) ≈ 6.908
        assert 4.5 < result["amt_log"].iloc[0] < 4.7
        assert 6.8 < result["amt_log"].iloc[1] < 7.0

    def test_gender_mapping(self):
        """Test gender binary encoding."""
        extractor = FraudFeatureExtractor()

        data = pd.DataFrame({"gender": ["M", "F", "M"]})

        result = extractor.transform(data)

        assert result["gender"].tolist() == [1, 0, 1]


class TestPipelineCreation:
    """Test pipeline factory function."""

    def test_create_pipeline(self):
        """Test that pipeline is created correctly."""
        params = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 50}

        pipeline = create_fraud_pipeline(params)

        # Check it's a valid estimator
        assert isinstance(pipeline, BaseEstimator)

        # Check steps exist
        assert "features" in pipeline.named_steps
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_pipeline_fit_predict(self):
        """Test that pipeline can fit and predict."""
        # Create minimal sample data
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "trans_date_trans_time": pd.date_range("2019-01-01", periods=n_samples, freq="H"),
                "amt": np.random.uniform(10, 500, n_samples),
                "lat": np.random.uniform(30, 45, n_samples),
                "long": np.random.uniform(-120, -70, n_samples),
                "merch_lat": np.random.uniform(30, 45, n_samples),
                "merch_long": np.random.uniform(-120, -70, n_samples),
                "job": np.random.choice(["Engineer, biomedical", "Data scientist"], n_samples),
                "category": np.random.choice(["grocery_pos", "gas_transport"], n_samples),
                "gender": np.random.choice(["M", "F"], n_samples),
                "dob": ["1990-01-01"] * n_samples,
                "trans_count_24h": np.random.randint(1, 10, n_samples),
                "amt_to_avg_ratio_24h": np.random.uniform(0.5, 2.0, n_samples),
                "amt_relative_to_all_time": np.random.uniform(0.5, 2.0, n_samples),
            }
        )

        y = np.random.randint(0, 2, n_samples)  # Random binary labels

        params = {"max_depth": 3, "n_estimators": 10}
        pipeline = create_fraud_pipeline(params)

        # Should fit without errors
        pipeline.fit(data, y)

        # Should predict
        predictions = pipeline.predict(data)
        assert len(predictions) == n_samples
        assert set(predictions).issubset({0, 1})

        # Should predict probabilities
        probas = pipeline.predict_proba(data)
        assert probas.shape == (n_samples, 2)
        assert np.all((probas >= 0) & (probas <= 1))

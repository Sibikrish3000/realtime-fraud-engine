"""
Tests for SHAP Explainability Engine.

Integration tests verifying SHAP functionality with trained models.
"""

import base64
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.explainability import FraudExplainer
from src.models.pipeline import create_fraud_pipeline


@pytest.fixture
def sample_transaction():
    """Create a valid sample transaction for testing."""
    return pd.DataFrame(
        [
            {
                "trans_date_trans_time": "2020-01-01 12:00:00",
                "amt": 150.0,
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 40.7200,
                "merch_long": -74.0100,
                "job": "Engineer, biomedical",
                "category": "grocery_pos",
                "gender": "M",
                "dob": "1990-01-01",
                "trans_count_24h": 3,
                "amt_to_avg_ratio_24h": 1.2,
                "amt_relative_to_all_time": 1.1,
            }
        ]
    )


@pytest.fixture
def trained_pipeline():
    """Create a minimal trained pipeline for testing."""
    # Create and quickly train a pipeline
    params = {"max_depth": 3, "n_estimators": 10, "learning_rate": 0.3}
    pipeline = create_fraud_pipeline(params)

    # Generate minimal training data
    np.random.seed(42)
    n_samples = 100

    X_train = pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range("2019-01-01", periods=n_samples, freq="h"),
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

    y_train = np.random.randint(0, 2, n_samples)

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        joblib.dump(pipeline, f.name)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


class TestFraudExplainer:
    """Test suite for FraudExplainer class."""

    def test_initialization(self, trained_pipeline):
        """Test that explainer initializes without errors."""
        explainer = FraudExplainer(trained_pipeline)

        assert explainer.pipeline is not None
        assert explainer.model is not None
        assert explainer.preprocessor is not None
        assert explainer.explainer is not None
        assert len(explainer.feature_names) > 0

    def test_initialization_invalid_path(self):
        """Test that explainer raises error for invalid path."""
        with pytest.raises(FileNotFoundError):
            FraudExplainer("/nonexistent/path.pkl")

    def test_generate_waterfall(self, trained_pipeline, sample_transaction):
        """Test waterfall plot generation for a single transaction."""
        explainer = FraudExplainer(trained_pipeline)

        # Generate waterfall (base64)
        waterfall_b64 = explainer.generate_waterfall(sample_transaction)

        # Verify it's a valid base64 string
        assert isinstance(waterfall_b64, str)
        assert len(waterfall_b64) > 0

        # Verify it decodes to valid bytes
        try:
            decoded = base64.b64decode(waterfall_b64)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Failed to decode base64: {e}")

    def test_generate_waterfall_multiple_transactions_fails(
        self, trained_pipeline, sample_transaction
    ):
        """Test that waterfall fails with multiple transactions."""
        explainer = FraudExplainer(trained_pipeline)

        # Create 2 transactions
        two_transactions = pd.concat([sample_transaction, sample_transaction])

        with pytest.raises(ValueError, match="Expected 1 transaction"):
            explainer.generate_waterfall(two_transactions)

    def test_generate_summary(self, trained_pipeline, sample_transaction):
        """Test summary plot generation for multiple transactions."""
        explainer = FraudExplainer(trained_pipeline)

        # Create sample of 10 transactions
        sample = pd.concat([sample_transaction] * 10, ignore_index=True)

        # Generate summary plot
        summary_b64 = explainer.generate_summary(sample)

        # Verify it's a valid base64 string
        assert isinstance(summary_b64, str)
        assert len(summary_b64) > 0

        # Verify it decodes
        try:
            decoded = base64.b64decode(summary_b64)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Failed to decode base64: {e}")

    def test_explain_prediction(self, trained_pipeline, sample_transaction):
        """Test comprehensive prediction explanation."""
        explainer = FraudExplainer(trained_pipeline)

        explanation = explainer.explain_prediction(sample_transaction, threshold=0.5)

        # Verify structure
        assert "prediction" in explanation
        assert "decision" in explanation
        assert "shap_values" in explanation
        assert "top_features" in explanation
        assert "base_value" in explanation

        # Verify types
        assert isinstance(explanation["prediction"], float)
        assert explanation["decision"] in ["BLOCK", "APPROVE"]
        assert isinstance(explanation["shap_values"], dict)
        assert isinstance(explanation["top_features"], list)
        assert len(explanation["top_features"]) == 5

        # Verify top_features structure
        for feature in explanation["top_features"]:
            assert "feature" in feature
            assert "impact" in feature
            assert "abs_impact" in feature

    def test_no_value_error_raised(self, trained_pipeline, sample_transaction):
        """Test that no ValueError is raised during normal operation."""
        explainer = FraudExplainer(trained_pipeline)

        # This should not raise ValueError
        try:
            waterfall = explainer.generate_waterfall(sample_transaction)
            summary = explainer.generate_summary(sample_transaction)
            explanation = explainer.explain_prediction(sample_transaction)
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError raised: {e}")

    def test_shap_values_calculation(self, trained_pipeline, sample_transaction):
        """Test SHAP value calculation."""
        explainer = FraudExplainer(trained_pipeline)

        shap_values, X_transformed = explainer.calculate_shap_values(sample_transaction)

        # Verify shapes
        assert shap_values.shape[0] == 1  # 1 transaction
        assert shap_values.shape[1] == len(explainer.feature_names)
        assert X_transformed.shape[0] == 1
        assert X_transformed.shape[1] == len(explainer.feature_names)

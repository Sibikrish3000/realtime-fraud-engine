"""
Integration tests for FastAPI inference service.
"""


import pytest
from fastapi.testclient import TestClient


# Note: These tests require a trained model to be present
# Run training first: python src/models/train.py --data_path ...


@pytest.fixture
def api_client():
    """Create test client for API."""
    # Import here to avoid startup issues before model is trained
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def sample_request_data():
    """Sample prediction request."""
    return {
        "user_id": "test_user_123",
        "trans_date_trans_time": "2020-06-15 14:30:00",
        "amt": 150.00,
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 40.7200,
        "merch_long": -74.0100,
        "job": "Engineer, biomedical",
        "category": "grocery_pos",
        "gender": "M",
        "dob": "1985-03-20",
    }


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_endpoint_exists(self, api_client):
        """Test that health endpoint is accessible."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, api_client):
        """Test health response has correct structure."""
        response = api_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "redis_connected" in data
        assert "version" in data


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @pytest.mark.skip(reason="Requires trained model - run after training")
    def test_predict_endpoint_returns_200(self, api_client, sample_request_data):
        """Test that predict endpoint returns 200 OK."""
        response = api_client.post("/v1/predict", json=sample_request_data)
        assert response.status_code == 200

    @pytest.mark.skip(reason="Requires trained model - run after training")
    def test_predict_response_structure(self, api_client, sample_request_data):
        """Test prediction response has correct structure."""
        response = api_client.post("/v1/predict", json=sample_request_data)
        data = response.json()

        # Required fields
        assert "decision" in data
        assert "probability" in data
        assert "risk_score" in data
        assert "latency_ms" in data
        assert "shadow_mode" in data

        # Value constraints
        assert data["decision"] in ["BLOCK", "APPROVE"]
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["risk_score"] <= 100
        assert data["latency_ms"] > 0

    @pytest.mark.skip(reason="Requires trained model - run after training")
    def test_latency_within_target(self, api_client, sample_request_data):
        """Test that latency is within 50ms target."""
        response = api_client.post("/v1/predict", json=sample_request_data)
        data = response.json()

        # Should be well under 50ms for single prediction
        assert data["latency_ms"] < 50.0

    def test_predict_invalid_request(self, api_client):
        """Test that invalid request returns 422 validation error."""
        invalid_data = {"amt": "not_a_number"}  # Invalid type
        response = api_client.post("/v1/predict", json=invalid_data)
        assert response.status_code == 422  # Unprocessable Entity


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


class TestShadowMode:
    """Tests for shadow mode functionality."""

    @pytest.mark.skip(reason="Requires trained model and shadow mode config")
    def test_shadow_mode_always_approves(self, api_client, sample_request_data):
        """Test that shadow mode always returns APPROVE."""
        # This test assumes shadow_mode=True in config
        response = api_client.post("/v1/predict", json=sample_request_data)
        data = response.json()

        if data["shadow_mode"]:
            assert data["decision"] == "APPROVE"

    @pytest.mark.skip(reason="Requires log file inspection")
    def test_shadow_mode_logs_predictions(self, api_client, sample_request_data, tmp_path):
        """Test that shadow mode logs predictions to file."""
        # Would need to inspect logs/shadow_predictions.jsonl
        # to verify logging occurred
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

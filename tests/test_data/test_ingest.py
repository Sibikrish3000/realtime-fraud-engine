"""
Tests for Data Ingestion Module

Tests Pydantic validation and dataset loading functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.ingest import (
    InferenceTransactionSchema,
    TransactionSchema,
    load_dataset,
)


class TestTransactionSchema:
    """Test suite for TransactionSchema validation."""

    def test_valid_transaction(self, sample_transaction):
        """Test that a valid transaction passes validation."""
        schema = TransactionSchema(**sample_transaction)
        assert schema.amt == 4.97
        assert schema.category == "misc_net"
        assert schema.is_fraud == 0

    def test_negative_amount_fails(self, sample_transaction):
        """Test that negative amounts are rejected."""
        sample_transaction["amt"] = -50.00
        with pytest.raises(ValidationError) as exc_info:
            TransactionSchema(**sample_transaction)
        assert "amt" in str(exc_info.value)

    def test_invalid_coordinates_fail(self, sample_transaction):
        """Test that invalid lat/long are rejected."""
        # Latitude out of range
        sample_transaction["lat"] = 200.0
        with pytest.raises(ValidationError) as exc_info:
            TransactionSchema(**sample_transaction)
        assert "lat" in str(exc_info.value)

        # Reset and test longitude
        sample_transaction["lat"] = 36.0788
        sample_transaction["long"] = -200.0
        with pytest.raises(ValidationError) as exc_info:
            TransactionSchema(**sample_transaction)
        assert "long" in str(exc_info.value)

    def test_invalid_category_fails(self, sample_transaction):
        """Test that unknown categories are rejected."""
        sample_transaction["category"] = "invalid_category"
        with pytest.raises(ValidationError) as exc_info:
            TransactionSchema(**sample_transaction)
        assert "category" in str(exc_info.value)

    def test_invalid_job_fails(self, sample_transaction):
        """Test that unknown jobs are rejected."""
        sample_transaction["job"] = "Invalid Job Title"
        with pytest.raises(ValidationError) as exc_info:
            TransactionSchema(**sample_transaction)
        assert "job" in str(exc_info.value)

    def test_invalid_timestamp_fails(self, sample_transaction):
        """Test that malformed timestamps are rejected."""
        sample_transaction["trans_date_trans_time"] = "invalid-timestamp"
        with pytest.raises(ValidationError) as exc_info:
            TransactionSchema(**sample_transaction)
        assert "timestamp" in str(exc_info.value).lower()


class TestInferenceTransactionSchema:
    """Test suite for InferenceTransactionSchema."""

    def test_valid_inference_request(self):
        """Test that a valid inference request passes validation."""
        data = {
            "user_id": "u12345",
            "amt": 150.00,
            "lat": 40.7128,
            "long": -74.0060,
            "category": "grocery_pos",
            "job": "Engineer, biomedical",
            "merch_lat": 40.7200,
            "merch_long": -74.0100,
            "unix_time": 1234567890,
        }
        schema = InferenceTransactionSchema(**data)
        assert schema.user_id == "u12345"
        assert schema.amt == 150.00

    def test_invalid_category_fails(self):
        """Test that invalid categories are rejected in inference."""
        data = {
            "user_id": "u12345",
            "amt": 150.00,
            "lat": 40.7128,
            "long": -74.0060,
            "category": "bad_category",
            "job": "Engineer, biomedical",
            "merch_lat": 40.7200,
            "merch_long": -74.0100,
            "unix_time": 1234567890,
        }
        with pytest.raises(ValidationError):
            InferenceTransactionSchema(**data)


class TestLoadDataset:
    """Test suite for load_dataset function."""

    def test_load_csv(self, sample_transaction):
        """Test loading a CSV file."""
        # Create temporary CSV
        df = pd.DataFrame([sample_transaction])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loaded_df = load_dataset(temp_path, validate=False)
            assert len(loaded_df) == 1
            assert loaded_df.iloc[0]["amt"] == 4.97
        finally:
            Path(temp_path).unlink()

    def test_load_with_validation(self, sample_transaction):
        """Test loading with validation enabled."""
        df = pd.DataFrame([sample_transaction])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loaded_df = load_dataset(temp_path, validate=True)
            assert len(loaded_df) == 1
        finally:
            Path(temp_path).unlink()

    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.csv")

    def test_unsupported_format(self):
        """Test that unsupported formats are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(ValueError) as exc_info:
                load_dataset(f.name)
            assert "Unsupported file format" in str(exc_info.value)
